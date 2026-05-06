# ============================================
# AutoArchitect — Agent Network
# Connects multiple agents into a collaborative
# intelligence network
#
# Every collaboration feeds the meta-learner
# Brain learns: which agents work best together
#               optimal network topology
#               collective intelligence patterns
#
# This is ANAS — Agent Network Architecture Search
# Nobody has published this yet.
# ============================================

import os
import json
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime

AGENTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))),
    'agent_data'
)


class AgentNetwork:
    """
    A network of collaborating autonomous agents.

    Agents in the network:
    → share knowledge automatically
    → hand off work to each other
    → collectively smarter than any single agent
    → feed collaboration data to meta-learner brain
    → brain learns optimal network topology

    Usage:
        network = AgentNetwork("pothole_system")
        network.add_agent(detection_agent)
        network.add_agent(severity_agent)
        network.add_agent(reporting_agent)
        network.run()
    """

    def __init__(self, name: str = None,
                  description: str = ""):
        self.network_id  = str(uuid.uuid4())[:8]
        self.name        = name or f"network_{self.network_id}"
        self.description = description
        self.agents      = {}      # agent_id → agent
        self.pipelines   = []      # ordered workflows
        self.is_running  = False
        self.created_at  = datetime.now().isoformat()

        # Network performance tracking
        self.total_runs        = 0
        self.successful_runs   = 0
        self.collaboration_log = []

        os.makedirs(AGENTS_DIR, exist_ok=True)

        from api.agents.fusion_agent import FusionAgent
        self.fusion_agent = FusionAgent()

        print(f"[Network] Agent Network [{self.network_id}] "
              f"created: {self.name}")

    # ── BUILD NETWORK ────────────────────────

    def add_agent(self, agent,
                   role: str = None) -> None:
        """Add an agent to the network.
        Works with both BaseAgent and legacy agents."""
        # Give legacy agents an agent_id if missing
        if not hasattr(agent, 'agent_id'):
            agent.agent_id = str(uuid.uuid4())[:8]
        if not hasattr(agent, 'category'):
            agent.category = role or 'image'
        if not hasattr(agent, 'problem'):
            agent.problem = role or 'unknown'
        if not hasattr(agent, 'total_predictions'):
            agent.total_predictions = 0
        if not hasattr(agent, '_get_memory_accuracy'):
            agent._get_memory_accuracy = lambda: 0.0
        if not hasattr(agent, 'run_async'):
            agent.run_async = lambda **kw: None
        if not hasattr(agent, 'stop'):
            agent.stop = lambda: None
        if not hasattr(agent, '_memory'):
            from api.agents.base_agent import AgentMemory
            agent._memory = AgentMemory(agent.agent_id)
        # act(result) — single-arg signature; ignore extra positional args
        if not hasattr(agent, 'act'):
            agent.act = lambda result, *_a, **_k: result
        # remember(input, prediction, action) — no-op for agents without memory
        if not hasattr(agent, 'remember'):
            agent.remember = lambda *_a, **_k: None

        agent_id = agent.agent_id
        self.agents[agent_id] = {
            "agent":  agent,
            "role":   role or agent.category,
            "added":  datetime.now().isoformat(),
        }
        print(f"  [Network+] Agent [{agent_id}] added to"
              f" [{self.network_id}]"
              f" as '{role or agent.category}'")

    def add_pipeline(self, agent_ids: list,
                      name: str = "") -> None:
        """
        Define a sequential pipeline of agents.
        Output of agent N becomes input of agent N+1.
        """
        self.pipelines.append({
            "name":      name or f"pipeline_{len(self.pipelines)}",
            "agents":    agent_ids,
            "created":   datetime.now().isoformat(),
        })
        print(f"  [Pipeline+] '{name}' added: "
              f"{' -> '.join(agent_ids)}")

    # ── RUN NETWORK ──────────────────────────

    def process(self, input_data,
                 pipeline_name: str = None) -> dict:
        """
        Run input through the agent network.
        If pipeline_name given → sequential pipeline
        Else → all agents vote (ensemble)
        Returns combined result + feeds brain.
        """
        start  = time.time()
        result = {}

        pipeline = self._get_pipeline(pipeline_name)

        if pipeline:
            result = self._run_pipeline(
                input_data, pipeline)
        else:
            result = self._run_ensemble(input_data)

        elapsed = round(time.time() - start, 3)

        # Feed network run to brain
        self._feed_brain_network_run(result, elapsed)

        self.total_runs += 1
        if result.get('success'):
            self.successful_runs += 1

        return result

    def run(self, source: str = None,
             interval: int = 60) -> None:
        """Run entire network autonomously."""
        self.is_running = True
        print(f"\n[Network] Network [{self.network_id}] running!")
        print(f"   Agents: {len(self.agents)}")
        print(f"   Source: {source or 'manual'}")

        # Start all agents in background
        threads = []
        for aid, a_info in self.agents.items():
            agent = a_info["agent"]
            t     = agent.run_async(
                interval=interval,
                source=source
            )
            threads.append(t)

        # Network-level collaboration loop
        try:
            while self.is_running:
                time.sleep(interval * 2)
                self._network_collaboration_cycle()
                self._network_status_report()
        except KeyboardInterrupt:
            self.stop()

    def run_async(self, source: str = None,
                   interval: int = 60
                   ) -> threading.Thread:
        """Run network in background thread."""
        t = threading.Thread(
            target=self.run,
            args=(source, interval),
            daemon=True
        )
        t.start()
        return t

    def stop(self) -> None:
        """Stop all agents in network."""
        self.is_running = False
        for aid, a_info in self.agents.items():
            a_info["agent"].stop()
        print(f"[Network] Network [{self.network_id}] stopped")

    # ── PIPELINE EXECUTION ───────────────────

    def _run_pipeline(self, input_data,
                       pipeline: dict) -> dict:
        """
        Run sequential pipeline.
        Each agent's output feeds next agent.
        """
        current_input  = input_data
        all_predictions = []
        pipeline_name  = pipeline["name"]

        print(f"\n  [Pipeline] Running pipeline: {pipeline_name}")

        for agent_id in pipeline["agents"]:
            if agent_id not in self.agents:
                print(f"  [Pipeline] Agent [{agent_id}] not in network -- skipping")
                continue

            agent      = self.agents[agent_id]["agent"]
            prediction = agent.predict(current_input)
            action     = agent.act(prediction)          # single-arg signature
            agent.remember(current_input, prediction, action)

            all_predictions.append({
                "agent_id":   agent_id,
                "role":       self.agents[agent_id]["role"],
                "prediction": prediction,
                "action":     action,
            })

            print(f"    [{agent_id}] -> "
                  f"{prediction.get('label')} "
                  f"({prediction.get('confidence')})")

            # Pass enriched context to next agent
            current_input = self._enrich_input(
                current_input, prediction)

        # Combine all predictions
        final = self._combine_predictions(all_predictions)
        final["pipeline"]  = pipeline_name
        final["steps"]     = len(all_predictions)
        final["success"]   = True

        # Log collaboration
        self._log_collaboration(all_predictions, final)

        return final

    def _run_ensemble(self, input_data) -> dict:
        """
        Run all agents in parallel, combine votes.
        Majority vote with confidence weighting.
        """
        print(f"\n  [Ensemble] Running ensemble vote...")
        all_predictions = []

        for aid, a_info in self.agents.items():
            agent = a_info["agent"]
            try:
                prediction = agent.predict(input_data)
                action     = agent.act(prediction)      # single-arg signature
                agent.remember(input_data, prediction, action)
                all_predictions.append({
                    "agent_id":   aid,
                    "role":       a_info["role"],
                    "prediction": prediction,
                    "action":     action,
                })
                print(f"    [{aid}] -> "
                      f"{prediction.get('label')} "
                      f"({prediction.get('confidence')})")
            except Exception as e:
                print(f"  [Ensemble] Agent [{aid}] failed: {e}")

        final = self._combine_predictions(all_predictions)
        final["ensemble"] = True
        final["votes"]    = len(all_predictions)
        final["success"]  = True

        self._log_collaboration(all_predictions, final)
        return final

    # ── COLLABORATE: parallel execution with timeouts and error isolation ────

    def collaborate(self, agents_list: list, task: str, data,
                    timeout_per_agent: int = 10, domain: str = None) -> dict:
        """
        Run multiple agents in parallel, fuse their outputs.
        Each agent gets timeout_per_agent seconds max.
        Skipped or failed agents don't block successful ones.

        Total wall time ≈ slowest successful agent (not sum of all agents).
        """
        submit_time = time.time()

        def run_agent(agent):
            start = time.time()
            try:
                result = agent.predict(data)
                result["agent_name"]     = agent.__class__.__name__
                result["execution_time"] = round(time.time() - start, 3)
                return result
            except Exception as e:
                return {
                    "agent_name":     agent.__class__.__name__,
                    "error":          str(e),
                    "confidence":     0.0,
                    "execution_time": round(time.time() - start, 3),
                }

        results = []

        # Submit all agents at once — real parallelism starts here.
        executor = ThreadPoolExecutor(max_workers=len(agents_list))
        try:
            future_to_agent = {
                executor.submit(run_agent, agent): agent
                for agent in agents_list
            }
            for future, agent in future_to_agent.items():
                # Remaining budget relative to batch submission so that
                # the ENTIRE batch completes within timeout_per_agent, not
                # each individual future.
                elapsed   = time.time() - submit_time
                remaining = max(0.01, timeout_per_agent - elapsed)
                try:
                    result = future.result(timeout=remaining)
                    results.append(result)
                    print(f"  [Collaborate] {agent.__class__.__name__} -> "
                          f"{result.get('label')} ({result.get('confidence')}) "
                          f"[{result['execution_time']}s]")
                except FutureTimeoutError:
                    results.append({
                        "agent_name":     agent.__class__.__name__,
                        "error":          "timeout",
                        "confidence":     0.0,
                        "execution_time": timeout_per_agent,
                    })
                    print(f"  [Collaborate] {agent.__class__.__name__} timed out "
                          f"(>{timeout_per_agent}s)")
        finally:
            # Don't block on threads that are still running after their
            # timeout budget expired — they continue as background threads.
            executor.shutdown(wait=False)

        valid_results = [r for r in results if "error" not in r]

        if not valid_results:
            return {
                "error":         "all agents failed",
                "details":       results,
                "fusion_method": "none",
                "task":          task,
                "agent_used":    "none",   # backward compat
            }

        fused = self.fusion_agent.fuse(valid_results, domain=domain)

        fused["task"]              = task
        fused["total_agents"]      = len(agents_list)
        fused["successful_agents"] = len(valid_results)
        fused["valid_count"]       = len(valid_results)   # backward compat
        fused["agent_count"]       = len(agents_list)     # backward compat
        fused["failed_agents"]     = len(results) - len(valid_results)
        fused["agent_timings"]     = {
            r["agent_name"]: r.get("execution_time", 0) for r in results
        }
        fused["wall_time"]         = round(time.time() - submit_time, 3)
        return fused

    # ── COLLABORATION CYCLE ──────────────────

    def _network_collaboration_cycle(self) -> None:
        """
        Periodic knowledge sharing between all agents.
        Every agent shares hard cases with every other.
        Brain learns which agent pairs collaborate best.
        """
        agents_list = list(self.agents.values())
        if len(agents_list) < 2:
            return

        print(f"\n  [Network] Collaboration cycle...")
        collab_results = []

        for i in range(len(agents_list)):
            for j in range(i+1, len(agents_list)):
                a1 = agents_list[i]["agent"]
                a2 = agents_list[j]["agent"]

                # Share hard cases both ways
                result = a1.collaborate(a2)
                collab_results.append(result)

                # Feed to brain
                self._feed_brain_collaboration(
                    a1, a2, result)

        print(f"  [Network] {len(collab_results)} collaborations complete")

    # ── COMBINE PREDICTIONS ──────────────────

    def _combine_predictions(self,
                              predictions: list) -> dict:
        """
        Combine multiple agent predictions into one.
        Uses confidence-weighted voting.
        """
        if not predictions:
            return {"label": "unknown",
                    "confidence": 0,
                    "success": False}

        if len(predictions) == 1:
            p = predictions[0]["prediction"]
            return {
                "label":      p.get("label", "unknown"),
                "confidence": p.get("confidence", 0),
                "action":     p.get("action", ""),
                "agent":      predictions[0]["agent_id"],
            }

        # Weighted vote by confidence
        vote_scores = {}
        for pred_info in predictions:
            p     = pred_info["prediction"]
            label = p.get("label", "unknown")
            conf  = p.get("confidence", 0)
            vote_scores[label] = \
                vote_scores.get(label, 0) + conf

        best_label = max(vote_scores,
                         key=vote_scores.get)
        best_conf  = round(
            vote_scores[best_label] / len(predictions), 1)

        # Get action from highest confidence agent
        best_action = ""
        best_single = max(
            predictions,
            key=lambda x: x["prediction"].get(
                "confidence", 0))
        best_action = best_single["prediction"].get(
            "action", "")

        return {
            "label":      best_label,
            "confidence": best_conf,
            "action":     best_action,
            "all_votes":  vote_scores,
            "n_agents":   len(predictions),
        }

    # ── BRAIN FEEDING ────────────────────────

    def _feed_brain_network_run(self, result: dict,
                                  elapsed: float) -> None:
        """Every network run teaches the brain."""
        try:
            from api.brain.meta_learner import get_meta_learner
            meta = get_meta_learner()

            agent_categories = [
                a["agent"].category
                for a in self.agents.values()
            ]

            meta.learn(
                problem         = self.name,
                agents_used     = agent_categories,
                dataset_used    = "agent_network_runtime",
                method_used     = "agent_network",
                actual_accuracy = result.get(
                    "confidence", 0),
            )
        except Exception:
            pass

    def _feed_brain_collaboration(self, a1, a2,
                                   result: dict) -> None:
        """Every collaboration teaches the brain."""
        try:
            from api.brain.meta_learner import get_meta_learner
            meta = get_meta_learner()
            meta.learn(
                problem         = f"{a1.problem} + {a2.problem}",
                agents_used     = [a1.category, a2.category],
                dataset_used    = "agent_collaboration",
                method_used     = "agent_network",
                actual_accuracy = (
                    a1._get_memory_accuracy() +
                    a2._get_memory_accuracy()) / 2,
            )
        except Exception:
            pass

    # ── HELPERS ──────────────────────────────

    def _get_pipeline(self,
                       name: str = None) -> dict:
        if not self.pipelines:
            return None
        if name:
            for p in self.pipelines:
                if p["name"] == name:
                    return p
        return self.pipelines[0]

    def _enrich_input(self, original_input,
                       prediction: dict):
        """Pass prediction context to next agent."""
        if isinstance(original_input, str):
            return original_input
        return {
            "input":      original_input,
            "context":    prediction,
        }

    def _log_collaboration(self, predictions: list,
                            final: dict) -> None:
        entry = {
            "timestamp":   datetime.now().isoformat(),
            "network_id":  self.network_id,
            "agents":      [p["agent_id"]
                            for p in predictions],
            "final":       final,
        }
        self.collaboration_log.append(entry)

        log_path = os.path.join(
            AGENTS_DIR,
            f"{self.network_id}_collab_log.jsonl")
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _network_status_report(self) -> None:
        print(f"\n  [Network] Status [{self.network_id}]:")
        print(f"     Agents:      {len(self.agents)}")
        print(f"     Total runs:  {self.total_runs}")
        print(f"     Success:     {self.successful_runs}")
        for aid, a_info in self.agents.items():
            agent = a_info["agent"]
            print(f"     [{aid}] {a_info['role']}: "
                  f"{agent.total_predictions} predictions, "
                  f"{agent._get_memory_accuracy()}% acc")

    def info(self) -> dict:
        return {
            "network_id":   self.network_id,
            "name":         self.name,
            "agents":       {
                aid: {
                    "role":        a["role"],
                    "category":    a["agent"].category,
                    "predictions": a["agent"].total_predictions,
                    "accuracy":    a["agent"]._get_memory_accuracy(),
                }
                for aid, a in self.agents.items()
            },
            "total_runs":   self.total_runs,
            "pipelines":    len(self.pipelines),
            "is_running":   self.is_running,
            "created_at":   self.created_at,
        }


# ============================================
# NETWORK BUILDER — creates network from problem
# ============================================

def build_network_from_problem(problem: str,
                                 agents_used: list,
                                 model_paths: dict = None
                                 ) -> AgentNetwork:
    """
    Auto-builds an agent network from a problem description.
    Called after AutoArchitect trains multi-agent pipeline.

    agents_used: ["image", "text", "security", ...]
    model_paths: {"image": "path/to/model.pth", ...}
    """
    from api.agents.image_agent       import ImageAgent
    from api.agents.text_agent        import TextAgent
    from api.agents.medical_agent     import MedicalAgent
    from api.agents.security_agent    import SecurityAgent
    from api.agents.tabular_agent     import TabularAgent
    from api.agents.audio_agent       import AudioAgent
    from api.agents.multimodal_agent  import MultimodalAgent

    network = AgentNetwork(
        name        = problem[:30],
        description = f"Auto-built network for: {problem}"
    )

    agent_classes = {
        "image":     ImageAgent,
        "text":      TextAgent,
        "medical":   MedicalAgent,
        "security":  SecurityAgent,
        "tabular":   TabularAgent,
        "audio":     AudioAgent,
        "multimodal": MultimodalAgent,
    }

    for domain in agents_used:
        if domain in agent_classes:
            # Create domain agent
            AgentClass = agent_classes[domain]
            agent = AgentClass()

            # Wrap with BaseAgent capabilities
            # by monkey-patching the predict method
            network.add_agent(agent, role=domain)

    # Add sequential pipeline
    if len(agents_used) > 1:
        network.add_pipeline(
            [a.agent_id for a in
             [network.agents[aid]["agent"]
              for aid in network.agents]],
            name="main_pipeline"
        )

    print(f"\n[Network] Agent network built for: {problem}")
    print(f"   Agents: {agents_used}")
    print(f"   Network ID: {network.network_id}")
    print(f"\nUsage:")
    print(f"  result = network.process('input.jpg')")
    print(f"  network.run(source='your_folder/')")

    return network
