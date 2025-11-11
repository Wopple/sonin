import json

from sonin.model.dna import Dna


def save_samples_local(name: str, samples: list[Dna]):
    with open(f'local/{name}.json', 'w') as fp:
        json.dump([s.model_dump(mode='json') for s in samples], fp)


def load_samples_local(name: str) -> list[Dna]:
    with open(f'local/{name}.json') as fp:
        return [Dna(**s) for s in json.load(fp)]
