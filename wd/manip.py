from clearml import Task
from clearml.backend_api.services.v2_20.tasks import MultiFieldPatternData
from clearml.backend_api.session.client import APIClient


def manip():
    client = APIClient()
    pattern = MultiFieldPatternData(pattern="debatable-pipit-72", fields=["name"])
    conf = client.tasks.get_all(_all_=pattern)
    print(conf)


if __name__ == '__main__':
    manip()