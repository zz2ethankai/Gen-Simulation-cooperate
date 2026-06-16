from __future__ import annotations

from eval.envs import create_env
from eval.policies import create_policy
from eval.runners.episode_runner import EpisodeRunner
from eval.specs import EvalSpec
from eval.storage import JsonlResultStore


class SuiteRunner:
    def __init__(self, spec: EvalSpec):
        self.spec = spec
        self.store = JsonlResultStore(spec)

    def run(self) -> dict:
        env = create_env(self.spec.task)
        policy = create_policy(self.spec.policy)
        results = []
        try:
            runner = EpisodeRunner(env, policy, self.spec, artifact_dir=self.store.run_dir)
            for seed in self.spec.seeds:
                print(f"[eval] running seed={seed}", flush=True)
                episode = runner.run(seed)
                self.store.write_episode(episode)
                results.append(episode)

            summary = self._summarize(results)
            summary["run_dir"] = str(self.store.run_dir)
            self.store.write_summary(summary)
            return summary
        except BaseException as exc:
            import traceback

            failure = {
                "eval_name": self.spec.name,
                "task": self.spec.task.name,
                "policy": self.spec.policy.name,
                "episodes": len(results),
                "successes": sum(1 for item in results if item["metrics"].get("success")),
                "success_rate": 0.0,
                "seeds": self.spec.seeds,
                "run_dir": str(self.store.run_dir),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            self.store.write_summary(failure)
            raise
        finally:
            policy.close()
            env.close()

    def _summarize(self, episodes: list[dict]) -> dict:
        total = len(episodes)
        successes = sum(1 for item in episodes if item["metrics"].get("success"))
        return {
            "eval_name": self.spec.name,
            "task": self.spec.task.name,
            "policy": self.spec.policy.name,
            "episodes": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "seeds": self.spec.seeds,
        }
