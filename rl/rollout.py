"""
Episode collection for REINFORCE training.

A rollout is one full agent episode where the RL policy makes a binary
{continue, stop} decision at each step. We record the log-probabilities of
each decision so REINFORCE can compute the policy gradient.

Key design:
  - The SFT LLM (frozen) handles all reasoning and tool calls.
  - The RL policy only decides WHEN to stop.
  - At each step BEFORE calling the LLM, the policy samples an action.
    * action=1 (stop): skip the LLM call, request a final answer, end episode.
    * action=0 (continue): call the LLM, execute the tool, continue.
  - This keeps the action space binary and the MDP well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from agent.agent import AgentResult, StepRecord, _parse_step, _is_correct, _extract_answer_from_raw, MAX_NEW_TOKENS, TEMPERATURE
from agent.tools import ToolExecutor
from rl.policy_network import PolicyNetwork, features_from_history
from rl.rewards import compute_reward_from_result, RewardBreakdown
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class Transition:
    """One step's worth of data for REINFORCE."""
    features:    torch.Tensor   # (1, 4) state features
    action:      int            # 0=continue, 1=stop
    log_prob:    torch.Tensor   # log π(action | state)


@dataclass
class Episode:
    """Complete episode: transitions + outcome."""
    question:     str
    gold_answer:  str
    transitions:  list[Transition] = field(default_factory=list)
    reward:       RewardBreakdown | None = None
    result:       AgentResult | None = None


class RolloutCollector:
    """
    Runs the agent with the RL policy making stop/continue decisions and
    records all (state, action, log_prob) transitions for REINFORCE.

    Args:
        sft_model      : frozen SFT LLM
        tokenizer      : matching tokenizer
        tool_executor  : ToolExecutor with pre-built index
        policy_network : trainable MLP
        max_steps      : hard cap on steps per episode
        alpha/beta/eps : reward hyperparameters
    """

    def __init__(
        self,
        sft_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        tool_executor: ToolExecutor,
        policy_network: PolicyNetwork,
        max_steps: int = 6,
        alpha: float = 0.1,
        beta:  float = 0.05,
        epsilon: float = 0.05,
    ):
        self.sft_model      = sft_model
        self.tokenizer      = tokenizer
        self.tool_executor  = tool_executor
        self.policy_network = policy_network
        self.max_steps      = max_steps
        self.alpha          = alpha
        self.beta           = beta
        self.epsilon        = epsilon
        self.device         = next(sft_model.parameters()).device

    def _build_prompt(self, question: str, history: list[StepRecord]) -> str:
        """Reuse agent prompt builder."""
        from data.sft_dataset import SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {question}"},
        ]
        base = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not history:
            return base

        import json
        lines = []
        for rec in history:
            step_json = {
                "thought":    rec.thought,
                "action":     rec.action,
                "confidence": rec.confidence,
            }
            if rec.action == "search":
                step_json["query"] = rec.tool_input
            elif rec.action == "read":
                step_json["document"] = rec.tool_input
            elif rec.action == "answer":
                step_json["answer"] = rec.answer
            lines.append(f"Step {rec.step_idx + 1}: {json.dumps(step_json)}")
            if rec.tool_output:
                lines.append(f"Observation: {rec.tool_output[:300]}")
        lines.append(f"Step {len(history) + 1}: ")
        return base + "\n".join(lines)

    @torch.no_grad()
    def _call_llm(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.sft_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def collect_episode(self, question: str, gold_answer: str) -> Episode:
        """
        Run one full episode, recording RL policy decisions.

        The policy is queried at the START of each step.  If it says stop,
        we ask the LLM for a final answer and end the episode immediately.
        """
        import json as _json

        episode = Episode(question=question, gold_answer=gold_answer)
        history: list[StepRecord] = []
        pred_answer  = ""
        stopped_by   = "max_steps"

        for step_idx in range(self.max_steps):
            # RL policy decision
            feats            = features_from_history(history).to(self.device)
            p_stop           = self.policy_network(feats).squeeze()
            dist             = torch.distributions.Bernoulli(p_stop)
            action_tensor    = dist.sample()
            log_prob         = dist.log_prob(action_tensor)
            action           = int(action_tensor.item())

            episode.transitions.append(Transition(
                features=feats.detach().cpu(),
                action=action,
                log_prob=log_prob,
            ))

            if action == 1:  # stop
                stopped_by = "policy"
                # Request final answer from LLM
                raw    = self._call_llm(self._build_prompt(question, history))
                parsed = _parse_step(raw)
                if parsed and parsed.get("action") == "answer":
                    pred_answer = parsed.get("answer", "")
                else:
                    pred_answer = _extract_answer_from_raw(raw)
                break

            # action == 0: continue — call LLM
            prompt     = self._build_prompt(question, history)
            raw_output = self._call_llm(prompt)
            parsed     = _parse_step(raw_output)

            if parsed is None:
                pred_answer = _extract_answer_from_raw(raw_output)
                stopped_by  = "parse_error"
                break

            act        = parsed.get("action", "answer")
            thought    = parsed.get("thought", "")
            confidence = float(parsed.get("confidence", 0.5))

            rec = StepRecord(
                step_idx=step_idx,
                action=act,
                thought=thought,
                confidence=confidence,
                raw_output=raw_output,
            )

            if act == "answer":
                rec.answer   = parsed.get("answer", "")
                pred_answer  = rec.answer
                stopped_by   = "answer"
                history.append(rec)
                break

            elif act == "search":
                query           = parsed.get("query", question)
                rec.tool_input  = query
                rec.tool_output = self.tool_executor.search(query)

            elif act == "read":
                doc             = parsed.get("document", "")
                rec.tool_input  = doc
                rec.tool_output = self.tool_executor.read(doc)

            history.append(rec)

        # Build AgentResult
        correct = _is_correct(pred_answer, gold_answer)
        result  = AgentResult(
            question=question,
            gold_answer=gold_answer,
            pred_answer=pred_answer,
            steps=history,
            n_steps=len(history),
            n_tool_calls=sum(1 for r in history if r.action in ("search", "read")),
            correct=correct,
            stopped_by=stopped_by,
        )

        reward = compute_reward_from_result(
            result, self.alpha, self.beta, self.epsilon
        )

        episode.result = result
        episode.reward = reward
        return episode

    def collect_batch(
        self, questions: list[dict], n_episodes: int | None = None
    ) -> list[Episode]:
        """Collect multiple episodes (one per question, up to n_episodes)."""
        qs = questions if n_episodes is None else questions[:n_episodes]
        episodes = []
        for q in qs:
            ep = self.collect_episode(q["question"], q.get("answer", ""))
            episodes.append(ep)
        return episodes
