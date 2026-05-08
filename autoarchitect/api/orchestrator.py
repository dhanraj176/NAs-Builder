# ============================================
# AutoArchitect — Universal Orchestrator v8
# Now with Self-Evaluator
# Brain evaluates its own output — no human needed
# Every network scored → brain learns → improves itself
# ============================================

import os
import io
import time
import requests
from api.analyzer        import ProblemAnalyzer
from api.workflow_engine import WorkflowEngine
from api.cache_manager   import (
    check_cache, save_to_cache,
    increment_use_count, find_similar_cached
)
from api.agents.dynamic_agent import DynamicAgent

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

LLM_KEYWORDS = [
    "write a", "write me", "write an",
    "tell me a story", "create a story",
    "debate about", "make a debate",
    "summarize this", "summarize the",
    "write an essay", "write a report",
    "give me an argument",
    "write a poem", "pros and cons of",
    "your opinion on", "explain the concept of",
]


class AutoArchitectOrchestrator:
    def __init__(self, groq_api_key: str = ""):
        print("[Orchestrator] Initializing AutoArchitect Orchestrator v8...")
        self.analyzer        = ProblemAnalyzer()
        self.workflow        = WorkflowEngine()
        self._agents         = {}
        self._last_embedding = None
        self.groq_key        = groq_api_key or os.getenv("GROQ_API_KEY", "")

        # ── Brain + Meta-Learner ──────────────────────────────────────────
        try:
            from api.brain.workflow_generator import WorkflowGenerator
            self.brain         = WorkflowGenerator()
            self.brain_enabled = True
            print("[Orchestrator] Brain enabled -- learning from every problem!")
        except Exception as e:
            print(f"[Orchestrator] Brain not loaded: {e}")
            self.brain         = None
            self.brain_enabled = False

        # ── Topology Designer ─────────────────────────────────────────────
        try:
            from api.brain.topology_designer import TopologyDesigner
            self.topology_designer = TopologyDesigner()
            self.topology_enabled  = True
            print("[Orchestrator] Topology Designer ready -- designing agent networks!")
        except Exception as e:
            print(f"[Orchestrator] Topology Designer not loaded: {e}")
            self.topology_designer = None
            self.topology_enabled  = False

        # ── Network Zip Generator ─────────────────────────────────────────
        try:
            from api.brain.network_zip_generator import NetworkZipGenerator
            self.network_zip         = NetworkZipGenerator()
            self.network_zip_enabled = True
            print("[Orchestrator] Network Zip Generator ready -- full networks to download!")
        except Exception as e:
            print(f"[Orchestrator] Network Zip Generator not loaded: {e}")
            self.network_zip         = None
            self.network_zip_enabled = False

        # ── Web Researcher ────────────────────────────────────────────────
        try:
            from api.brain.web_researcher import WebResearcher
            self.researcher         = WebResearcher(groq_api_key=self.groq_key)
            self.researcher_enabled = True
            print("[Orchestrator] Web Researcher ready -- brain searches internet!")
        except Exception as e:
            print(f"[Orchestrator] Web Researcher not loaded: {e}")
            self.researcher         = None
            self.researcher_enabled = False

        # ── Self Evaluator (NEW) ──────────────────────────────────────────
        try:
            from api.brain.self_evaluator import SelfEvaluator
            self.self_evaluator         = SelfEvaluator()
            self.self_evaluator_enabled = True
            print("[Orchestrator] Self Evaluator ready -- brain evaluates its own output!")
        except Exception as e:
            print(f"[Orchestrator] Self Evaluator not loaded: {e}")
            self.self_evaluator         = None
            self.self_evaluator_enabled = False

        # ── Output Generator ─────────────────────────────────────────────
        try:
            from api.brain.output_generator import generate_output
            self._generate_output = generate_output
            self.output_enabled   = True
            print("[Orchestrator] Output Generator enabled -- human readable results!")
        except Exception as e:
            print(f"[Orchestrator] Output generator not loaded: {e}")
            self._generate_output = None
            self.output_enabled   = False

        # Store last workflow result for network generation
        self._last_workflow_result = {}
        self._last_problem         = ""
        self._last_topology        = {}
        self._last_research        = {}
        self._last_eval_result     = {}

        print("[Orchestrator] v8 ready -- NAS + Topology + Network + Research + Self-Eval!")

    # ─────────────────────────────────────────────────────────────────────
    # PUBLIC — main entry point
    # ─────────────────────────────────────────────────────────────────────
    def solve(self, problem: str, image_data: str = "", progress_callback=None) -> dict:
        start = time.time()

        # 1. LLM check first
        if self._needs_llm(problem):
            analysis = self.analyzer.analyze(problem)
            result   = self._run_llm(problem)
            result.update({
                "domain":   "text_generation",
                "analysis": analysis,
                "elapsed":  round(time.time() - start, 2),
            })
            return result

        # 2. Classify with BERT
        analysis = self.analyzer.analyze(problem)
        domain   = analysis["category"]
        conf     = analysis["confidence"]
        # Fix BERT domain before brain picks agents
        try:
          from api.self_trainer import _correct_domain
          corrected = _correct_domain(problem, domain)
          if corrected != domain:
                print(f"[Orchestrator] Early domain correction: {domain} -> {corrected}")
                domain = corrected
                analysis["category"] = domain
        except Exception:
          pass

        # 3. Get BERT embedding for meta-learner
        try:
            from api.cache_manager import get_embedding
            self._last_embedding = get_embedding(problem)
        except Exception:
            self._last_embedding = None

        # 4. Cache check
        cached = check_cache(problem)
        if cached["found"]:
            print(f"[Orchestrator] Cache hit: {problem[:40]}")
            increment_use_count(cached)
            meta        = cached["metadata"]
            result_type = meta.get("result_type", "single")
            agents_used = meta.get("agents_used", [domain])
            all_acc     = meta.get("all_accuracies", {})
            # avg_accuracy (multi-agent) or test_accuracy (self_trainer) —
            # use whichever is populated; default 0 only when both absent.
            avg_acc     = meta.get("avg_accuracy") or meta.get("test_accuracy") or 0

            base = {
                "status":       "success",
                "from_cache":   True,
                "domain":       domain,
                "analysis":     analysis,
                "architecture": meta.get("architecture", []),
                "parameters":   meta.get("parameters", 0),
                "search_time":  meta.get("search_time", 0),
                "use_count":    meta.get("use_count", 1),
                "elapsed":      round(time.time() - start, 2),
                "message":      "Loaded instantly from knowledge base!",
                "type":         result_type,
                "agents_used":  agents_used,
                "evaluation":   meta.get("evaluation", {}),
                "train_size":   meta.get("train_size", 0),
                "test_size":    meta.get("test_size", 0),
                "real_dataset": meta.get("real_dataset", False),
                "dataset":      meta.get("dataset", ""),
                "test_accuracy": avg_acc,
                "avg_accuracy":  avg_acc,
            }

            if result_type == "multi_agent_nas":
                base["self_trained"]   = meta.get("self_trained", False)
                base["avg_accuracy"]   = avg_acc
                base["all_accuracies"] = all_acc

            base["readable_output"] = self._get_readable_output(problem, base)
            base["topology"]        = self._design_topology(
                problem, domain, base, agents_used)

            if self.brain_enabled:
                try:
                    self.brain.learn_from_result(
                        problem        = problem,
                        workflow       = {"strategy_name": result_type,
                                          "agents": agents_used},
                        accuracy       = avg_acc,
                        time_taken     = round(time.time() - start, 2),
                        dataset_used   = meta.get("dataset", "unknown"),
                        method_used    = meta.get("method", "darts_nas"),
                        bert_embedding = self._last_embedding,
                        from_cache     = True
                    )
                except Exception as e:
                    print(f"[Orchestrator] Brain update skipped: {e}")

            self._last_workflow_result = base
            self._last_problem         = problem
            return base

        # 5. Similar problem check
        similar = find_similar_cached(problem, domain)
        if similar:
            print(f"[Orchestrator] Similar: {similar['problem'][:40]}")
            sim_meta        = similar.get("metadata", {})
            sim_result_type = sim_meta.get("result_type", "single")
            sim_agents_used = sim_meta.get("agents_used", [domain])
            sim_avg_acc     = sim_meta.get("avg_accuracy") or sim_meta.get("test_accuracy") or 0

            base = {
                "status":       "success",
                "from_cache":   True,
                "domain":       domain,
                "analysis":     analysis,
                "architecture": sim_meta.get("architecture", []),
                "parameters":   sim_meta.get("parameters", 0),
                "search_time":  sim_meta.get("search_time", 0),
                "use_count":    sim_meta.get("use_count", 1),
                "elapsed":      round(time.time() - start, 2),
                "message":      "Loaded instantly from knowledge base!",
                "type":         sim_result_type,
                "agents_used":  sim_agents_used,
                "evaluation":   sim_meta.get("evaluation", {}),
                "train_size":   sim_meta.get("train_size", 0),
                "test_size":    sim_meta.get("test_size", 0),
                "real_dataset": sim_meta.get("real_dataset", False),
                "dataset":      sim_meta.get("dataset", ""),
                "test_accuracy": sim_avg_acc,
                "avg_accuracy":  sim_avg_acc,
            }
            base["readable_output"] = self._get_readable_output(problem, base)
            base["topology"]        = self._design_topology(
                problem, domain, base, sim_agents_used)
            self._last_workflow_result = base
            self._last_problem         = problem
            return base

        # 6. Brain generates optimal workflow
        if self.brain_enabled:
            try:
                workflow = self.brain.generate(
                    problem,
                    domain,
                    bert_embedding=self._last_embedding
                )
                print(f"[Orchestrator] Brain workflow: {workflow['type']} "
                      f"-- {workflow['agents']} "
                      f"(strategy: {workflow['strategy_name']}, "
                      f"source: {workflow.get('source', 'unknown')})")
            except Exception as e:
                print(f"[Orchestrator] Brain failed, fallback: {e}")
                workflow = self.workflow.build_workflow(problem, domain)
        else:
            workflow = self.workflow.build_workflow(problem, domain)

        print(f"[Orchestrator] Workflow: {workflow['type']} -- agents: {workflow['agents']}")

        # 7. Web Research — find best approach before running agents
        research = {}
        if self.researcher_enabled:
            try:
                research = self.researcher.research(
                    problem = problem,
                    domain  = domain,
                )
                self._last_research = research
                print(f"[Orchestrator] Research: {research.get('best_model','?')} | "
                      f"Dataset: {research.get('best_dataset','?')} | "
                      f"Expected: {research.get('expected_acc','?')}")
            except Exception as e:
                print(f"[Orchestrator] Research skipped: {e}")

        # 8. Run agents
        if progress_callback:
            progress_callback(0, 6, 'Dataset search starting', phase='Dataset search')
        if workflow["type"] == "multi":
            # Parallel when brain says parallel/hybrid, or by default for multi-agent
            brain_res = workflow.get("brain_result") or {}
            exec_mode = brain_res.get("architecture", {}).get(
                "execution_mode", "parallel")
            if exec_mode in ("parallel", "hybrid"):
                result = self._run_parallel_ensemble(
                    problem, workflow["agents"], image_data,
                    progress_callback=progress_callback)
            else:
                result = self._run_multi_agent(
                    problem, workflow["agents"], image_data,
                    progress_callback=progress_callback)
        else:
            result = self._run_single_agent(
                problem, domain, image_data,
                progress_callback=progress_callback)

        # 9. Design topology
        agents_used        = result.get("agents_used", [domain])
        topology           = self._design_topology(problem, domain, result, agents_used)
        result["topology"] = topology

        # 10. Save to cache
        save_to_cache(
            problem        = problem,
            category       = domain,
            confidence     = conf,
            architecture   = result.get("architecture", []),
            parameters     = result.get("parameters", 0),
            search_time    = result.get("search_time", 0),
            result_type    = result.get("type", "single"),
            agents_used    = agents_used,
            self_trained   = result.get("self_trained", False),
            avg_accuracy   = result.get("avg_accuracy", 0),
            all_accuracies = result.get("all_accuracies", {}),
            evaluation     = result.get("evaluation", {}),
            test_size      = result.get("test_size", 0),
            real_dataset   = result.get("real_dataset", False),
            dataset        = result.get("dataset", ""),
            train_size     = result.get("train_size", 0),
        )

        # 11. Brain learns from result
        if self.brain_enabled:
            try:
                accuracy = result.get("avg_accuracy",
                           result.get("test_accuracy", 0))
                self.brain.learn_from_result(
                    problem         = problem,
                    workflow        = workflow,
                    accuracy        = accuracy,
                    time_taken      = round(time.time() - start, 2),
                    dataset_used    = result.get("dataset", "unknown"),
                    method_used     = result.get("method", "darts_nas"),
                    bert_embedding  = self._last_embedding,
                    from_cache      = False
                )
                if self.topology_enabled and accuracy:
                    self.topology_designer.update_accuracy(problem, accuracy / 100)
                if accuracy:
                    try:
                        from api.agents.fusion_agent import refresh_fusion_weights
                        refresh_fusion_weights()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[Orchestrator] Brain learn skipped: {e}")

        # 12. Generate human readable output
        result["readable_output"] = self._get_readable_output(problem, result)
        result.update({
            "domain":           domain,
            "analysis":         analysis,
            "workflow":         workflow,
            "elapsed":          round(time.time() - start, 2),
            "brain_strategy":   workflow.get("strategy_name", ""),
            "brain_confidence": workflow.get("confidence", 0),
            "brain_source":     workflow.get("source", "unknown"),
            "research":         research,
        })

        self._last_workflow_result = result
        self._last_problem         = problem
        return result

    # ─────────────────────────────────────────────────────────────────────
    # NETWORK GENERATION — brain builds + evaluates its own output
    # ─────────────────────────────────────────────────────────────────────
    def generate_network_zip(self, problem: str = None,
                              workflow_result: dict = None) -> tuple:
        """
        Generate complete multi-agent network zip.
        Brain self-evaluates the zip it just built.
        Returns (zip_bytes, topology, eval_result)
        """
        problem         = problem or self._last_problem
        workflow_result = workflow_result or self._last_workflow_result

        if not self.topology_enabled or not self.network_zip_enabled:
            return self._fallback_single_agent_zip(problem, workflow_result), {}, {}

        meta_suggestion = None
        if self.brain_enabled:
            try:
                meta_suggestion = self.brain.meta_suggest(problem) \
                    if hasattr(self.brain, 'meta_suggest') else None
            except Exception:
                pass

        domain   = workflow_result.get("domain", "text")
        topology = self.topology_designer.design(
            problem         = problem,
            domain          = domain,
            meta_suggestion = meta_suggestion,
        )
        self._last_topology = topology

        print(f"\n[Orchestrator] Network topology:")
        print(f"   Agents:    {' -> '.join(topology['agents'])}")
        print(f"   Topology:  {topology['topology']}")
        print(f"   Confidence:{topology['confidence']:.0%}")
        print(f"   Source:    {topology['source']}")

        trained_models = {}
        for agent_result in workflow_result.get("agent_results", []):
            atype = agent_result.get("domain", agent_result.get("agent_type", ""))
            mpath = agent_result.get("model_path", "")
            if atype and mpath:
                trained_models[atype] = mpath

        # Generate the zip
        zip_bytes = self.network_zip.generate(
            problem        = problem,
            topology       = topology,
            trained_models = trained_models,
        )

        # ── Brain self-evaluates the zip it just built ────────────────────
        eval_result = {}
        if self.self_evaluator_enabled:
            try:
                eval_result = self.self_evaluator.evaluate(
                    zip_bytes = zip_bytes,
                    problem   = problem,
                    topology  = topology,
                    domain    = domain,
                )
                print(f"[Orchestrator] Self-eval: {eval_result['score']}/100 "
                      f"({eval_result['grade']}) -- "
                      f"{'PASSED' if eval_result['passed'] else 'needs improvement'}")

                # Feed self-eval score back to topology brain
                # Brain now knows quality of topology -- not just training accuracy
                if self.topology_enabled:
                    self.topology_designer.update_accuracy(
                        problem,
                        eval_result["score"] / 100
                    )
            except Exception as e:
                print(f"[Orchestrator] Self-eval skipped: {e}")

        self._last_eval_result = eval_result

        # Update accuracy in brain
        accuracy = workflow_result.get("avg_accuracy",
                   workflow_result.get("test_accuracy", 0))
        if accuracy and not eval_result:
            self.topology_designer.update_accuracy(problem, accuracy / 100)

        # Feed to meta-learner
        if self.brain_enabled:
            try:
                self.brain.learn_from_result(
                    problem        = problem,
                    workflow       = {
                        "strategy_name": f"network_{topology['topology']}",
                        "agents":        topology["agents"],
                    },
                    accuracy       = eval_result.get("score", accuracy) if eval_result else accuracy,
                    time_taken     = 0,
                    dataset_used   = workflow_result.get("dataset", "unknown"),
                    method_used    = "network_generation",
                    bert_embedding = self._last_embedding,
                    from_cache     = False,
                )
            except Exception as e:
                print(f"[Orchestrator] Brain network update skipped: {e}")

        return zip_bytes, topology, eval_result

    def _fallback_single_agent_zip(self, problem: str, result: dict) -> bytes:
        try:
            from api.brain.agent_generator import AgentGenerator
            gen = AgentGenerator()
            return gen.generate(problem, result)
        except Exception as e:
            print(f"[Orchestrator] Fallback zip failed: {e}")
            return b""

    # ─────────────────────────────────────────────────────────────────────
    # TOPOLOGY DESIGN HELPER
    # ─────────────────────────────────────────────────────────────────────
    def _design_topology(self, problem: str, domain: str,
                          result: dict, agents_used: list) -> dict:
        if not self.topology_enabled:
            return {}
        try:
            topology = self.topology_designer.design(
                problem = problem,
                domain  = domain,
            )
            print(f"[ANAS] Selected topology: {topology.get('topology','?')}  "
                  f"agents={topology.get('agents',[])}  "
                  f"conf={topology.get('confidence',0):.0%}  "
                  f"source={topology.get('source','?')}", flush=True)
            return topology
        except Exception as e:
            print(f"[Orchestrator] Topology design skipped: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────────────
    # READABLE OUTPUT HELPER
    # ─────────────────────────────────────────────────────────────────────
    def _get_readable_output(self, problem: str, result: dict) -> dict:
        if self.output_enabled:
            try:
                output = self._generate_output(
                    problem  = problem,
                    result   = result,
                    groq_key = self.groq_key
                )
                print(f"  [Orchestrator] Output: {output.get('verdict')} "
                      f"({output.get('overall_score')}%)")
                return output
            except Exception as e:
                print(f"[Orchestrator] Output generation skipped: {e}")

        eval_score = 0
        evaluation = result.get("evaluation", {})
        if evaluation:
            eval_score = evaluation.get("avg_score", 0)

        return {
            "overall_score":   eval_score or 75,
            "verdict":         "Good",
            "summary":         f"AutoArchitect analyzed: {problem}",
            "findings":        ["✅ Analysis complete"],
            "recommendations": ["Upload your own data for better accuracy"],
            "next_steps":      "Test with your real data",
            "confidence":      "Medium",
            "generated_by":    "AutoArchitect",
            "problem":         problem
        }

    # ─────────────────────────────────────────────────────────────────────
    # SINGLE AGENT PIPELINE
    # ─────────────────────────────────────────────────────────────────────
    def _run_single_agent(self, problem: str,
                           domain: str, image_data: str,
                           progress_callback=None) -> dict:
        print(f"[Orchestrator] Single agent: {domain}")
        if progress_callback:
            progress_callback(1, 6, 'Agent waking up', phase='NAS search', epoch=0, total_epochs=3)
        agent  = self._wake_agent(domain,problem)
        result = agent.run(problem, image_data)

        try:
            print(f"[AGENT] {domain} training started...", flush=True)
            from api.self_trainer import self_train
            _t0 = time.time()
            trained = self_train(problem=problem, category=domain, epochs=3,
                                 progress_callback=progress_callback)
            _elapsed = round(time.time() - _t0, 1)
            result["self_trained"]   = True
            result["train_accuracy"] = trained["train_accuracy"]
            result["test_accuracy"]  = trained["test_accuracy"]
            result["dataset"]        = trained["dataset"]
            result["train_size"]     = trained["train_size"]
            result["test_size"]      = trained.get("test_size", 0)
            result["real_dataset"]   = trained.get("real_dataset", False)
            result["method"]         = trained.get("method", "darts_nas")
            result["real_training"]  = True
            result["model_path"]     = trained.get("model_path")
            result["classes"]        = trained.get("classes", [])
            print(f"[AGENT] {domain} training complete: "
                  f"{trained['test_accuracy']}% in {_elapsed}s  "
                  f"dataset={trained['dataset']}", flush=True)

            # Connect trained model to agent via factory
            mp      = trained.get("model_path")
            classes = trained.get("classes", [])
            if mp and classes:
                try:
                    from api.agents.agent_factory import get_factory
                    factory    = get_factory()
                    new_agent  = factory.create_from_trained(
                        problem, domain, trained)
                    key = f"{domain}_{problem[:20]}"
                    self._agents[key] = new_agent
                    print(f"   [Orchestrator] {new_agent.class_name} "
                          f"connected to trained model")
                    result["agent_name"]  = new_agent.agent_name
                    result["class_name"]  = new_agent.class_name
                except Exception as e:
                    print(f"   [Orchestrator] Agent connection: {e}")

        except Exception as e:
            print(f"[Orchestrator] Self-train skipped: {e}")
            result["self_trained"] = False


        evaluator  = self._wake_evaluator()
        evaluation = evaluator.evaluate(result, problem)
        self._sleep_agent("evaluator")
        self._sleep_agent(domain ,  problem)

        result["type"]         = "single_agent_nas"
        result["agents_used"]  = [domain]
        result["evaluation"]   = evaluation
        result["avg_accuracy"] = result.get("test_accuracy", 0)
        result["message"]      = (
            f"✅ {domain.upper()} NAS + Self-Training complete! "
            f"Score: {evaluation['avg_score']}%"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────
    # PARALLEL ENSEMBLE PIPELINE
    # ─────────────────────────────────────────────────────────────────────
    def _run_parallel_ensemble(self, problem: str,
                                domains: list, image_data: str,
                                progress_callback=None) -> dict:
        """
        Train all domain agents simultaneously using ThreadPoolExecutor,
        then combine their outputs via FusionAgent with learned weights.

        Both models are saved as models/trained/{hash}_{domain}.pth so the
        NetworkZipGenerator can package them together.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        import api.self_trainer as _st

        _lock = threading.Lock()

        def train_domain(domain: str) -> dict:
            agent  = self._wake_agent(domain, problem)
            result = agent.run(problem, image_data)
            result["domain"] = domain

            try:
                print(f"[AGENT] {domain} training started...", flush=True)
                _t0 = time.time()
                # Reset deduplication so each domain can use any dataset
                with _lock:
                    _st._used_datasets_this_run = set()
                trained = _st.agent.train(problem, domain, epochs=3)
                _elapsed = round(time.time() - _t0, 1)

                result.update({
                    "train_accuracy": trained["train_accuracy"],
                    "test_accuracy":  trained["test_accuracy"],
                    "dataset":        trained["dataset"],
                    "train_size":     trained["train_size"],
                    "method":         trained.get("method", "darts_nas"),
                    "self_trained":   True,
                    "model_path":     trained.get("model_path"),
                    "classes":        trained.get("classes", []),
                })
                print(f"[AGENT] {domain} training complete: "
                      f"{trained['test_accuracy']}% in {_elapsed}s  "
                      f"dataset={trained['dataset']}", flush=True)
            except Exception as e:
                print(f"[AGENT] {domain} training failed: {e}", flush=True)
                result["self_trained"] = False

            return result

        print(f"[Orchestrator] Parallel ensemble: {domains}")

        agent_results  = []
        all_accuracies = []
        all_acc_dict   = {}

        with ThreadPoolExecutor(max_workers=len(domains)) as executor:
            futures = {
                executor.submit(train_domain, d): d for d in domains
            }
            for future in as_completed(futures):
                domain = futures[future]
                try:
                    res = future.result()
                    agent_results.append(res)
                    if res.get("test_accuracy"):
                        all_accuracies.append(res["test_accuracy"])
                        all_acc_dict[domain] = res["test_accuracy"]
                except Exception as e:
                    print(f"  [Parallel] {domain} future failed: {e}")

        # Restore original domain order
        _order = {d: i for i, d in enumerate(domains)}
        agent_results.sort(
            key=lambda r: _order.get(r.get("domain", ""), 99))

        # Connect trained agents so factory wires models
        trained_agents = []
        for res in agent_results:
            mp  = res.get("model_path")
            cls = res.get("classes", [])
            dom = res.get("domain")
            if mp and cls:
                try:
                    from api.agents.agent_factory import get_factory
                    ta  = get_factory().create_from_trained(
                        problem, dom, res)
                    key = f"{dom}_{problem[:20]}"
                    self._agents[key] = ta
                    trained_agents.append(ta)
                    res["agent_name"] = ta.agent_name
                    res["class_name"] = getattr(ta, "class_name", dom)
                except Exception as e:
                    print(f"   [Parallel] agent connect {dom}: {e}")

        # Ensemble inference via AgentNetwork.collaborate()
        ensemble_inference = {}
        if len(trained_agents) >= 2 and image_data:
            try:
                from api.agents.agent_network import AgentNetwork
                net = AgentNetwork(
                    name=f"ensemble_{problem[:20]}")
                ensemble_inference = net.collaborate(
                    trained_agents, problem, image_data,
                    domain=domains[0])
                print(f"  [Parallel] Ensemble: "
                      f"{ensemble_inference.get('label')} "
                      f"conf={ensemble_inference.get('confidence')}")
            except Exception as e:
                print(f"  [Parallel] Ensemble inference skipped: {e}")

        # Fuse architectures + evaluate
        from api.agents.fusion_agent import FusionAgent
        fused = FusionAgent().fuse_architectures(agent_results, problem)

        evaluator  = self._wake_evaluator()
        evaluation = evaluator.evaluate(fused, problem)
        self._sleep_agent("evaluator")
        for d in domains:
            self._sleep_agent(d, problem)

        avg_accuracy = round(
            sum(all_accuracies) / len(all_accuracies), 1
        ) if all_accuracies else 0

        return {
            "status":             "success",
            "type":               "parallel_ensemble_nas",
            "agents_used":        domains,
            "agent_results":      agent_results,
            "architecture":       fused.get("architecture", []),
            "parameters":         fused.get("parameters", 0),
            "search_time":        fused.get("search_time", 0),
            "fusion":             fused,
            "evaluation":         evaluation,
            "self_trained":       len(all_accuracies) > 0,
            "avg_accuracy":       avg_accuracy,
            "all_accuracies":     all_acc_dict,
            "ensemble_inference": ensemble_inference,
            "dataset":            (agent_results[0].get("dataset", "unknown")
                                   if agent_results else "unknown"),
            "method":             (agent_results[0].get("method", "darts_nas")
                                   if agent_results else "darts_nas"),
            "message": (
                f"Parallel Ensemble NAS complete! "
                f"{len(domains)} agents trained simultaneously. "
                f"Score: {evaluation['avg_score']}%"
                + (f" | Accuracy: {avg_accuracy}%" if avg_accuracy else "")
            ),
        }

    # ─────────────────────────────────────────────────────────────────────
    # MULTI AGENT PIPELINE (sequential — kept for sequential topologies)
    # ─────────────────────────────────────────────────────────────────────
    def _run_multi_agent(self, problem: str,
                          domains: list, image_data: str,
                          progress_callback=None) -> dict:
        print(f"[Orchestrator] Multi-agent pipeline: {domains}")

        agent_results  = []
        all_accuracies = []
        all_acc_dict   = {}

        for domain in domains:
            print(f"  [Orchestrator] Running {domain} NAS agent...")
            agent  = self._wake_agent(domain)
            result = agent.run(problem, image_data)
            result["domain"] = domain

            try:
                print(f"  [Orchestrator] Auto self-training {domain}...")
                from api.self_trainer import self_train
                trained = self_train(problem=problem,
                                     category=domain, epochs=3)
                result["train_accuracy"] = trained["train_accuracy"]
                result["test_accuracy"]  = trained["test_accuracy"]
                result["dataset"]        = trained["dataset"]
                result["train_size"]     = trained["train_size"]
                result["method"]         = trained.get("method", "darts_nas")
                result["self_trained"]   = True
                result["model_path"]     = trained.get("model_path")
                result["classes"]        = trained.get("classes", [])
                all_accuracies.append(trained["test_accuracy"])
                all_acc_dict[domain]     = trained["test_accuracy"]
                print(f"  [Orchestrator] {domain} self-trained: "
                      f"{trained['test_accuracy']}%")

                # Connect trained model to agent
                mp      = trained.get("model_path")
                classes = trained.get("classes", [])
                if mp and classes:
                    try:
                        from api.agents.agent_factory import get_factory
                        new_agent = get_factory().create_from_trained(
                            problem, domain, trained)
                        key = f"{domain}_{problem[:20]}"
                        self._agents[key] = new_agent
                        print(f"   [Orchestrator] {new_agent.class_name} connected")
                        result["agent_name"]  = new_agent.agent_name
                        result["class_name"]  = new_agent.class_name
                    except Exception as e:
                        print(f"   [Orchestrator] Agent connection: {e}")

            except Exception as e:
                print(f"  [Orchestrator] {domain} self-train skipped: {e}")
                result["self_trained"] = False

            agent_results.append(result)
            self._sleep_agent(domain)

        print("  [Orchestrator] Fusing architectures...")
        from api.agents.fusion_agent import FusionAgent
        fusion = FusionAgent()
        fused  = fusion.fuse_architectures(agent_results, problem)

        print("  [Orchestrator] Evaluating fused model...")
        evaluator  = self._wake_evaluator()
        evaluation = evaluator.evaluate(fused, problem)
        self._sleep_agent("evaluator")

        avg_accuracy = round(
            sum(all_accuracies) / len(all_accuracies), 1
        ) if all_accuracies else 0

        return {
            "status":         "success",
            "type":           "multi_agent_nas",
            "agents_used":    domains,
            "agent_results":  agent_results,
            "architecture":   fused.get("architecture", []),
            "parameters":     fused.get("parameters", 0),
            "search_time":    fused.get("search_time", 0),
            "fusion":         fused,
            "evaluation":     evaluation,
            "self_trained":   len(all_accuracies) > 0,
            "avg_accuracy":   avg_accuracy,
            "all_accuracies": all_acc_dict,
            "dataset":        agent_results[0].get("dataset", "unknown") if agent_results else "unknown",
            "method":         agent_results[0].get("method", "darts_nas") if agent_results else "darts_nas",
            "message": (
                f"🔀 Multi-Agent NAS complete! "
                f"{len(domains)} agents fused. "
                f"Score: {evaluation['avg_score']}%"
                + (f" | Accuracy: {avg_accuracy}%" if avg_accuracy else "")
            )
        }

    # ─────────────────────────────────────────────────────────────────────
    # LAZY LOADING
    # ─────────────────────────────────────────────────────────────────────
    def _wake_agent(self, domain: str, problem: str = ""):
        key = f"{domain}_{problem[:20]}" if problem else domain
        if key not in self._agents:
            print(f"  [Orchestrator] Loading {domain} agent for: {problem[:30]}")
            from api.agents.agent_factory import get_factory
            factory = get_factory()
            agent   = factory.create(
                problem = problem or domain,
                domain  = domain,
            )
            self._agents[key] = agent
        return self._agents[key]

    def _sleep_agent(self, domain: str, problem: str = ""):
       key = f"{domain}_{problem[:20]}" if problem else domain
       if key in self._agents:
         print(f"  [Orchestrator] Unloading {domain} agent...")
         del self._agents[key]
       elif domain in self._agents:
         print(f"  [Orchestrator] Unloading {domain} agent...")
         del self._agents[domain]


    def _wake_evaluator(self):
        if "evaluator" not in self._agents:
            from api.agents.evaluator_agent import EvaluatorAgent
            self._agents["evaluator"] = EvaluatorAgent()
        return self._agents["evaluator"]


    # ─────────────────────────────────────────────────────────────────────
    # LLM
    # ─────────────────────────────────────────────────────────────────────
    def _needs_llm(self, problem: str) -> bool:
        return any(kw in problem.lower() for kw in LLM_KEYWORDS)

    def _run_llm(self, problem: str) -> dict:
        print(f"[Orchestrator] LLM: {problem[:40]}")
        if not self.groq_key:
            return {
                "status":  "success",
                "type":    "llm_generation",
                "output":  self._fallback_llm(problem),
                "model":   "fallback",
                "message": "Add GROQ_API_KEY for LLM generation.",
            }
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type":  "application/json",
            }
            payload = {
                "model":    GROQ_MODEL,
                "messages": [
                    {
                        "role":    "system",
                        "content": (
                            "You are AutoArchitect AI — a helpful assistant "
                            "that solves any problem clearly and concisely. "
                            "Give structured, practical responses."
                        ),
                    },
                    {"role": "user", "content": problem},
                ],
                "max_tokens":  800,
                "temperature": 0.7,
            }
            resp = requests.post(
                GROQ_API_URL, json=payload,
                headers=headers, timeout=30
            )
            resp.raise_for_status()
            data   = resp.json()
            output = data["choices"][0]["message"]["content"]
            return {
                "status":  "success",
                "type":    "llm_generation",
                "output":  output,
                "model":   GROQ_MODEL,
                "message": "Generated by Llama 3 via Groq",
            }
        except Exception as e:
            print(f"[Orchestrator] Groq error: {e}")
            return {
                "status":  "success",
                "type":    "llm_generation",
                "output":  self._fallback_llm(problem),
                "model":   "fallback",
                "message": f"Groq unavailable: {str(e)}",
            }

    def _fallback_llm(self, problem: str) -> str:
        return (
            f"AutoArchitect received: '{problem}'\n\n"
            "Add GROQ_API_KEY to .env for full LLM generation."
        )
