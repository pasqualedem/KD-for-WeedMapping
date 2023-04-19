from clearml.backend_api.session.client import APIClient
import pandas as pd


# Create an instance of APIClient
client = APIClient()
project_list = client.projects.get_all(name="example*")


def get_by_project_name(project_name):
    project_list = client.projects.get_all(name=project_name)
    return project_list
    
def get_df_by_project_id(project_ids):
    tasks = client.tasks.get_all(project=project_ids)
    hp = client.tasks.get_hyper_params(tasks=[task.id for task in tasks])
    hp = get_hp_as_pd(hp)
    
    tasks_more = [client.tasks.get_by_id(task.id) for task in tasks]
    metrics = [{"task": t.id, "metrics": t.last_metrics} for t in tasks_more]
    metrics = get_metrics_as_pd(metrics)
    return pd.merge(hp, metrics, on='task', how='inner')


def get_metrics_as_pd(metrics):
    flattener = lambda d: {'task': d['task'], **{value_dict['variant']: value_dict['value']
             for subdict in d['metrics'].values()
             for value_dict in subdict.values()}}
    return pd.DataFrame(map(flattener, metrics))


def get_hp_as_pd(hp):
    df = pd.DataFrame(hp.params)

    # Extract hyperparameter names and values into new columns
    df = pd.concat([df.drop('hyperparams', axis=1), df['hyperparams'].apply(lambda x: pd.Series({d['name']: d['value'] for d in x}))], axis=1)
    return df
