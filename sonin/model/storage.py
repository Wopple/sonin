import json

from sonin.model.mutation import DnaMutagen


def save_samples_local(name: str, samples: list[DnaMutagen]):
    with open(f'local/{name}.json', 'w') as fp:
        json.dump([s.model_dump(mode='json') for s in samples], fp)


def load_samples_local(name: str) -> list[DnaMutagen]:
    with open(f'local/{name}.json') as fp:
        return [DnaMutagen(**s) for s in json.load(fp)]
