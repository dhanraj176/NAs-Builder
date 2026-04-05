# ============================================
# AutoArchitect — Base Agent
# Universal interface for ALL autonomous agents
#
# Every agent that AutoArchitect generates
# inherits from this class.
#
# An agent can:
#   perceive()    → watch for new input
#   predict()     → run trained model
#   act()         → take real-world action
#   remember()    → store in persistent memory
#   learn()       → retrain on accumulated memory
#   collaborate() → share knowledge with other agents
#   run()         → loop forever autonomously
#
# Every action feeds back to the meta-learner brain
# → brain gets smarter with every agent interaction
# ============================================

import os
import time
import json
import uuid
import threading
from datetime import datetime
from abc      import ABC, abstractmethod

AGENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))),
    'agent_data'
)


class BaseAgent(ABC):
    """
    Universal autonomous agent base class.
    Every AutoArchitect-generated agent inherits this.

    The agent loop:
    perceive → predict → act → remember → learn → repeat
    """

    def __init__(self,
                 agent_id:   str  = None,
                 problem:    str  = "",
                 category:   str  = "image",
                 classes:    list = None,
                 accuracy:   float = 0.0,
                 config:     dict  = None):

        self.agent_id   = agent_id or str(uuid.uuid4())[:8]
        self.problem    = problem
        self.category   = category
        self.classes    = classes or []
        self.accuracy   = accuracy
        self.config     = config or {}

        # Agent state
        self.is_running   = False
        self.is_trained   = False
        self.total_predictions = 0
        self.correct_predictions = 0
        self.created_at   = datetime.now().isoformat()

        # Actions registry
        # {"action_name": callable}
        self._actions     = {}

        # Collaboration registry
        # {"agent_id": BaseAgent}
        self._network     = {}

        # Memory system
        self._memory      = AgentMemory(self.agent_id)

        # Setup agent data directory
        os.makedirs(AGENTS_DIR, exist_ok=True)

        print(f"🤖 Agent [{self.agent_id}] initialized")
        print(f"   Problem:  {self.problem[:40]}")
        print(f"   Category: {self.category}")
        print(f"   Classes:  {self.classes}")

    # ── CORE METHODS (must implement) ────────

    @abstractmethod
    def predict(self, input_data) -> dict:
        """
        Run prediction on input.
        Must return: {label, confidence, action}
        """
        pass

    # ── PERCEIVE ─────────────────────────────

    def perceive(self, source: str = None) -> list:
        """
        Watch for new inputs from source.
        source: folder path, API endpoint, or None
        Returns list of new inputs to process.
        """
        source = source or self.config.get('input_source')
        if not source:
            return []

        inputs = []

        # Watch folder for new files
        if os.path.isdir(source):
            processed = self._memory.get_processed_files()
            for fname in os.listdir(source):
                fpath = os.path.join(source, fname)
                if fpath not in processed:
                    if fname.lower().endswith(
                            ('.jpg','.jpeg','.png',
                             '.txt','.csv','.json')):
                        inputs.append(fpath)

        print(f"  👁️  Agent [{self.agent_id}] perceived "
              f"{len(inputs)} new inputs")
        return inputs

    # ── ACT ──────────────────────────────────

    def act(self, prediction: dict,
             input_data = None) -> dict:
        """
        Take real-world action based on prediction.
        Runs all registered action handlers.
        Returns action results.
        """
        results  = {}
        label    = prediction.get('label', '')
        conf     = prediction.get('confidence', 0)

        for name, handler in self._actions.items():
            try:
                result       = handler(prediction, input_data)
                results[name] = result
                print(f"  ⚡ Agent [{self.agent_id}] "
                      f"action '{name}': {result}")
            except Exception as e:
                results[name] = {'error': str(e)}

        # Default action — log everything
        self._log_action(prediction, results)
        return results

    def register_action(self, name: str,
                         handler) -> None:
        """Register a callable action handler."""
        self._actions[name] = handler
        print(f"  ✅ Agent [{self.agent_id}] "
              f"registered action: {name}")

    # ── REMEMBER ─────────────────────────────

    def remember(self, input_data,
                  prediction: dict,
                  action_result: dict = None,
                  ground_truth: str   = None) -> None:
        """
        Store interaction in persistent memory.
        Every memory entry feeds the meta-learner.
        """
        entry = {
            "agent_id":      self.agent_id,
            "problem":       self.problem,
            "category":      self.category,
            "label":         prediction.get('label'),
            "confidence":    prediction.get('confidence'),
            "correct":       (prediction.get('label') ==
                              ground_truth) if ground_truth
                             else None,
            "action_taken":  list(action_result.keys())
                             if action_result else [],
            "timestamp":     datetime.now().isoformat(),
        }

        self._memory.store(entry)
        self.total_predictions += 1

        if ground_truth and entry['correct']:
            self.correct_predictions += 1

        # Feed to meta-learner brain
        self._feed_brain(entry)

        print(f"  🧠 Agent [{self.agent_id}] remembered: "
              f"{entry['label']} ({entry['confidence']}%)")

    # ── LEARN ────────────────────────────────

    def learn(self, min_examples: int = 20) -> bool:
        """
        Retrain on accumulated memory.
        Called automatically every N predictions.
        Returns True if retraining happened.
        """
        memories = self._memory.get_all()

        if len(memories) < min_examples:
            print(f"  📚 Agent [{self.agent_id}] "
                  f"needs {min_examples - len(memories)} "
                  f"more examples to learn")
            return False

        # Extract training examples from memory
        correct   = [m for m in memories
                     if m.get('correct') is True]
        incorrect = [m for m in memories
                     if m.get('correct') is False]

        print(f"  🔄 Agent [{self.agent_id}] learning...")
        print(f"     Total memories:   {len(memories)}")
        print(f"     Correct:          {len(correct)}")
        print(f"     Incorrect:        {len(incorrect)}")
        print(f"     Memory accuracy:  "
              f"{round(len(correct)/max(len(memories),1)*100,1)}%")

        # Trigger meta-learner update
        self._update_meta_learner_from_memory(memories)

        return True

    # ── COLLABORATE ──────────────────────────

    def collaborate(self, other_agent,
                     input_data = None) -> dict:
        """
        Share knowledge with another agent.
        Both agents improve from the exchange.
        Feeds collaboration data to meta-learner.
        """
        other_id = other_agent.agent_id

        print(f"  🤝 Agent [{self.agent_id}] collaborating "
              f"with [{other_id}]")

        # Share my memories with other agent
        my_memories    = self._memory.get_all()
        other_memories = other_agent._memory.get_all()

        # Find examples the other agent hasn't seen
        shared = 0
        for mem in my_memories:
            if mem.get('label') in other_agent.classes:
                other_agent._memory.store({
                    **mem,
                    "source_agent": self.agent_id,
                    "shared":       True
                })
                shared += 1

        # Record collaboration in meta-learner
        collab_result = {
            "agent_1":        self.agent_id,
            "agent_2":        other_id,
            "examples_shared": shared,
            "my_accuracy":    self._get_memory_accuracy(),
            "their_accuracy": other_agent._get_memory_accuracy(),
            "timestamp":      datetime.now().isoformat(),
        }

        self._feed_brain_collaboration(collab_result)

        print(f"  ✅ Shared {shared} examples with "
              f"agent [{other_id}]")
        return collab_result

    def join_network(self, network) -> None:
        """Join an agent network."""
        self._network[network.network_id] = network
        network.add_agent(self)
        print(f"  🔗 Agent [{self.agent_id}] "
              f"joined network [{network.network_id}]")

    # ── RUN ──────────────────────────────────

    def run(self, interval: int = 60,
             source: str = None,
             max_iterations: int = None) -> None:
        """
        Run agent autonomously forever.
        interval: seconds between perception cycles
        source:   input source (folder/API)
        max_iterations: None = run forever
        """
        self.is_running = True
        iterations      = 0

        print(f"\n🚀 Agent [{self.agent_id}] running!")
        print(f"   Problem:  {self.problem}")
        print(f"   Interval: {interval}s")
        print(f"   Source:   {source or 'manual'}")
        print(f"   Press Ctrl+C to stop\n")

        try:
            while self.is_running:
                if (max_iterations and
                        iterations >= max_iterations):
                    break

                print(f"\n{'='*40}")
                print(f"  Cycle {iterations+1} — "
                      f"{datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*40}")

                # Full agent loop
                inputs = self.perceive(source)

                for inp in inputs:
                    # Predict
                    prediction = self.predict(inp)

                    # Act
                    action_result = self.act(
                        prediction, inp)

                    # Remember
                    self.remember(
                        inp, prediction, action_result)

                    # Mark as processed
                    self._memory.mark_processed(inp)

                # Learn every 20 predictions
                if self.total_predictions > 0 and \
                   self.total_predictions % 20 == 0:
                    self.learn()

                # Status report
                self._status_report()

                iterations += 1
                if self.is_running:
                    time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n⏹️  Agent [{self.agent_id}] stopped")
            self.is_running = False

    def run_async(self, interval: int = 60,
                   source: str = None) -> threading.Thread:
        """Run agent in background thread."""
        t = threading.Thread(
            target=self.run,
            args=(interval, source),
            daemon=True
        )
        t.start()
        print(f"  🔄 Agent [{self.agent_id}] "
              f"running in background")
        return t

    def stop(self) -> None:
        """Stop the agent."""
        self.is_running = False
        print(f"  ⏹️  Agent [{self.agent_id}] stopping...")

    # ── INFO ─────────────────────────────────

    def info(self) -> dict:
        return {
            "agent_id":    self.agent_id,
            "problem":     self.problem,
            "category":    self.category,
            "classes":     self.classes,
            "accuracy":    self.accuracy,
            "is_running":  self.is_running,
            "is_trained":  self.is_trained,
            "predictions": self.total_predictions,
            "memory_size": len(self._memory.get_all()),
            "actions":     list(self._actions.keys()),
            "network_connections": len(self._network),
            "memory_accuracy": self._get_memory_accuracy(),
            "created_at":  self.created_at,
        }

    # ── PRIVATE HELPERS ──────────────────────

    def _feed_brain(self, memory_entry: dict) -> None:
        """Feed every prediction to meta-learner."""
        try:
            from api.brain.meta_learner import get_meta_learner
            meta = get_meta_learner()

            # Only feed if we have ground truth
            if memory_entry.get('correct') is not None:
                acc = (self._get_memory_accuracy()
                       if self.total_predictions > 5
                       else self.accuracy)
                meta.learn(
                    problem         = self.problem,
                    agents_used     = [self.category],
                    dataset_used    = "agent_runtime",
                    method_used     = "agent_operational",
                    actual_accuracy = acc,
                )
        except Exception as e:
            pass  # never block agent for brain update

    def _feed_brain_collaboration(self,
                                   collab: dict) -> None:
        """Feed collaboration data to meta-learner."""
        try:
            from api.brain.meta_learner import get_meta_learner
            meta = get_meta_learner()
            # Store as special collaboration example
            meta.learn(
                problem         = f"collaboration: "
                                  f"{collab['agent_1']}+"
                                  f"{collab['agent_2']}",
                agents_used     = [self.category],
                dataset_used    = "agent_collaboration",
                method_used     = "agent_network",
                actual_accuracy = (
                    collab['my_accuracy'] +
                    collab['their_accuracy']) / 2,
            )
        except Exception:
            pass

    def _update_meta_learner_from_memory(self,
                                          memories: list):
        """Bulk update meta-learner from memory."""
        try:
            from api.brain.meta_learner import get_meta_learner
            meta      = get_meta_learner()
            acc       = self._get_memory_accuracy()
            meta.learn(
                problem         = self.problem,
                agents_used     = [self.category],
                dataset_used    = "agent_memory_bulk",
                method_used     = "continual_learning",
                actual_accuracy = acc,
            )
            print(f"  🧠 Meta-learner updated from "
                  f"{len(memories)} memories "
                  f"({acc}% accuracy)")
        except Exception as e:
            print(f"  ⚠️ Meta-learner update failed: {e}")

    def _get_memory_accuracy(self) -> float:
        memories = self._memory.get_all()
        labeled  = [m for m in memories
                    if m.get('correct') is not None]
        if not labeled:
            return self.accuracy
        correct = sum(1 for m in labeled
                      if m.get('correct'))
        return round(correct / len(labeled) * 100, 1)

    def _log_action(self, prediction: dict,
                     results: dict) -> None:
        log_path = os.path.join(
            AGENTS_DIR, f"{self.agent_id}_actions.jsonl")
        entry = {
            "timestamp":  datetime.now().isoformat(),
            "prediction": prediction,
            "results":    results,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _status_report(self) -> None:
        print(f"\n  📊 Agent [{self.agent_id}] Status:")
        print(f"     Predictions: {self.total_predictions}")
        print(f"     Memory acc:  "
              f"{self._get_memory_accuracy()}%")
        print(f"     Memories:    "
              f"{len(self._memory.get_all())}")


# ============================================
# AGENT MEMORY — persistent per-agent storage
# ============================================

class AgentMemory:
    """
    Persistent memory for one agent.
    Every prediction stored → feeds brain.
    """

    def __init__(self, agent_id: str):
        self.agent_id  = agent_id
        self.mem_file  = os.path.join(
            AGENTS_DIR, f"{agent_id}_memory.json")
        self.proc_file = os.path.join(
            AGENTS_DIR, f"{agent_id}_processed.json")
        os.makedirs(AGENTS_DIR, exist_ok=True)
        self._memories   = self._load(self.mem_file)
        self._processed  = self._load(self.proc_file,
                                       default=[])

    def store(self, entry: dict) -> None:
        self._memories.append(entry)
        self._save(self.mem_file, self._memories)

    def get_all(self) -> list:
        return self._memories

    def get_recent(self, n: int = 100) -> list:
        return self._memories[-n:]

    def mark_processed(self, filepath: str) -> None:
        self._processed.append(filepath)
        self._save(self.proc_file, self._processed)

    def get_processed_files(self) -> set:
        return set(self._processed)

    def get_hard_cases(self,
                        conf_threshold: float = 60.0
                        ) -> list:
        """Examples agent was uncertain about."""
        return [m for m in self._memories
                if m.get('confidence', 100) < conf_threshold]

    def get_mistakes(self) -> list:
        """Examples agent got wrong."""
        return [m for m in self._memories
                if m.get('correct') is False]

    def stats(self) -> dict:
        total   = len(self._memories)
        labeled = [m for m in self._memories
                   if m.get('correct') is not None]
        correct = sum(1 for m in labeled
                      if m.get('correct'))
        return {
            "total":     total,
            "labeled":   len(labeled),
            "correct":   correct,
            "accuracy":  round(
                correct / max(len(labeled), 1) * 100, 1),
            "hard_cases": len(self.get_hard_cases()),
            "mistakes":   len(self.get_mistakes()),
        }

    def _load(self, path: str,
               default = None) -> list:
        if default is None:
            default = []
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save(self, path: str, data) -> None:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
