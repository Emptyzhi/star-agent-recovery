"""Explicit HTTP control plane for the selected resident."""
import requests


def exchange(endpoint, operation, payload):
    if operation=='health':
        response=requests.get(endpoint+'/health',timeout=30)
    elif operation in ('begin_run','end_run'):
        response=requests.post(endpoint+'/'+operation,json=payload,timeout=30)
    else:
        raise ValueError('unknown HTTP control operation: '+operation)
    response.raise_for_status()
    return response.json()
