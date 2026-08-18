"""Bounded parallel evolution over typed scene-layout candidates."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Any, Callable, Iterable

from .mutations import SceneMutation, _mutation_dict


@dataclass(frozen=True)
class SearchConfig:
    population_size: int = 8
    max_generations: int = 5
    worker_count: int = 4
    debug_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    heldout_seeds: tuple[int, ...] = tuple(range(100, 120))

    def __post_init__(self) -> None:
        if self.population_size != 8 or self.max_generations != 5:
            raise ValueError("the v1 evolution budget is fixed at K=8 and G=5")
        if self.worker_count != 4:
            raise ValueError("the v1 scheduler has four logical worker queues")
        if self.debug_seeds != (0, 1, 2, 3, 4):
            raise ValueError("debug seeds are frozen at 0-4")
        if self.heldout_seeds != tuple(range(100, 120)):
            raise ValueError("held-out seeds are frozen at 100-119")


@dataclass(frozen=True)
class CandidateGenome:
    candidate_id: str
    generation: int
    scene_revision: str
    profile_id: str
    parent_id: str | None = None
    mutations: tuple[SceneMutation, ...] = ()
    skill_parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mutations"] = [_mutation_dict(mutation) for mutation in self.mutations]
        return value


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    generation: int
    seed: int
    hard_constraints: dict[str, bool]
    planning_success: bool
    collision_margin_m: float = 0.0
    reach_margin_m: float = 0.0
    occlusion_score: float = 1.0
    path_length_m: float = float("inf")
    diversity_score: float = 0.0
    failure_code: str | None = None
    artifact_refs: tuple[str, ...] = ()

    @property
    def hard_ok(self) -> bool:
        return bool(self.hard_constraints) and all(self.hard_constraints.values())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateAggregate:
    genome: CandidateGenome
    evaluations: tuple[CandidateEvaluation, ...]

    @property
    def robust_success(self) -> bool:
        return bool(self.evaluations) and all(
            evaluation.hard_ok and evaluation.planning_success
            for evaluation in self.evaluations
        )

    @property
    def ranking_key(self) -> tuple[float, ...]:
        if not self.evaluations:
            return (0.0,) * 8
        count = float(len(self.evaluations))
        hard_pass_rate = sum(evaluation.hard_ok for evaluation in self.evaluations) / count
        planning_rate = sum(evaluation.planning_success for evaluation in self.evaluations) / count
        collision_margin = min(evaluation.collision_margin_m for evaluation in self.evaluations)
        reach_margin = min(evaluation.reach_margin_m for evaluation in self.evaluations)
        occlusion = sum(evaluation.occlusion_score for evaluation in self.evaluations) / count
        path_length = sum(evaluation.path_length_m for evaluation in self.evaluations) / count
        diversity = sum(evaluation.diversity_score for evaluation in self.evaluations) / count
        return (
            float(hard_pass_rate == 1.0),
            hard_pass_rate,
            planning_rate,
            collision_margin,
            reach_margin,
            -occlusion,
            -path_length,
            diversity,
        )


Evaluator = Callable[[CandidateGenome, int, int], CandidateEvaluation]
Evolver = Callable[[tuple[CandidateAggregate, ...], int, int], Iterable[CandidateGenome]]


@dataclass(frozen=True)
class EvolutionResult:
    generations_completed: int
    ranked: tuple[CandidateAggregate, ...]
    robust_winner: CandidateGenome | None


class EvolutionSearch:
    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()

    def run(
        self,
        initial_population: Iterable[CandidateGenome],
        evaluator: Evaluator,
        evolver: Evolver,
        artifact_dir: Path,
    ) -> EvolutionResult:
        population = tuple(initial_population)
        self._validate_population(population, 0)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        ranked: tuple[CandidateAggregate, ...] = ()
        for generation in range(self.config.max_generations):
            evaluations = self._evaluate(population, evaluator)
            self._write_generation(artifact_dir, generation, evaluations)
            by_candidate = {genome.candidate_id: [] for genome in population}
            for evaluation in evaluations:
                by_candidate[evaluation.candidate_id].append(evaluation)
            ranked = tuple(
                sorted(
                    (
                        CandidateAggregate(
                            genome,
                            tuple(sorted(by_candidate[genome.candidate_id], key=lambda item: item.seed)),
                        )
                        for genome in population
                    ),
                    key=lambda item: item.ranking_key,
                    reverse=True,
                )
            )
            winner = next((item.genome for item in ranked if item.robust_success), None)
            if winner is not None:
                return EvolutionResult(generation + 1, ranked, winner)
            if generation + 1 == self.config.max_generations:
                break
            population = tuple(
                evolver(ranked, generation + 1, self.config.population_size)
            )
            self._validate_population(population, generation + 1)
        return EvolutionResult(self.config.max_generations, ranked, None)

    def _evaluate(
        self,
        population: tuple[CandidateGenome, ...],
        evaluator: Evaluator,
    ) -> tuple[CandidateEvaluation, ...]:
        slots: SimpleQueue[int] = SimpleQueue()
        for slot in range(self.config.worker_count):
            slots.put(slot)

        def evaluate(genome: CandidateGenome, seed: int) -> CandidateEvaluation:
            slot = slots.get()
            try:
                result = evaluator(genome, seed, slot)
            finally:
                slots.put(slot)
            if result.candidate_id != genome.candidate_id or result.generation != genome.generation:
                raise ValueError("evaluator returned an evaluation for a different candidate")
            if result.seed != seed:
                raise ValueError("evaluator returned an evaluation for a different seed")
            return result

        futures = []
        with ThreadPoolExecutor(max_workers=self.config.worker_count) as executor:
            for genome in population:
                for seed in self.config.debug_seeds:
                    futures.append(executor.submit(evaluate, genome, seed))
            results = [future.result() for future in as_completed(futures)]
        return tuple(sorted(results, key=lambda item: (item.candidate_id, item.seed)))

    def _validate_population(
        self,
        population: tuple[CandidateGenome, ...],
        generation: int,
    ) -> None:
        if len(population) != self.config.population_size:
            raise ValueError(
                f"generation {generation} requires {self.config.population_size} candidates"
            )
        identifiers = [candidate.candidate_id for candidate in population]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"generation {generation} contains duplicate candidate ids")
        if any(candidate.generation != generation for candidate in population):
            raise ValueError(f"every candidate must declare generation {generation}")

    @staticmethod
    def _write_generation(
        artifact_dir: Path,
        generation: int,
        evaluations: tuple[CandidateEvaluation, ...],
    ) -> None:
        path = artifact_dir / f"generation_{generation:02d}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for evaluation in evaluations:
                stream.write(json.dumps(evaluation.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
