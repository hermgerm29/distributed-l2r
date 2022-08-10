# distributed-l2r
Achieving near-human speed performance in l2r with Kubernetes

## Example Usage

For brevity, some parts of this example, such as Dockerfiles and requirement installation, are excluded.

### Worker

The worker should be running in a Docker image that has the Arrival simulator already installed. The `AsyncWorker` class simply collects data in the environment and sends it to the learner. The worker needs little configuration. On initialization, an `AsyncWorker` will send a request to the learner for the policy it should use to collect data.

Here is an example of a script that you would run in the worker pods:

```python
import socket
from distrib_l2r.asynchron.worker import AsnycWorker

# This assumes that your kubernetes configuration has a service
# with the name `learner` and is using the default port 4444
learner_ip = socket.gethostbyname("learner")
learner_address = (learner_ip, 4444)

# Replace this with a gym wrapper so that your agent is
# exposed to processed data (e.g. encode images in the wrapper)
env_wrapper = <your_wrapper> 

if __name__ == '__main__':
	worker = AsnycWorker(learner_address=learner_address, env_wrapper=env_wrapper)
	worker.work()
```

### Learner

This base configuration uses a single learning node which maintains the policy and replay buffer along with performing the actual gradient updates. There are other ways of performing distributed RL that have currently not been implemented yet. The learner is non-blocking which slightly increases off-policy error but dramatically increases peak throughput - there should be no issues running a few dozen workers in parallel.

```python
from distrib_l2r.asynchron.learner import AsyncLearningNode

# Here you should create a policy and potentially load saved weights
policy = <your_policy>

if __name__ == '__main__':
	learner = AsyncLearningNode(policy=policy)
```

### Kubernetes Configuration

GPU sharing is not trivial with K8's. In order to distribute workers to avoid Cuda memory issues, it is recommended that you use `TBD` to create a configuration file to your specification.

With that said, the basic template of what this system looks like is shown below. One learner service serves sets of worker pods which use an environment variable to specify the GPU that they are using. Running more than 3 workers on the same GPU will likely cause errors.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: learner
spec:
  nodeSelector:
    nodetype: phortx
  containers:
    - name: learner-container
      tty: true
      stdin: true
      resources:
        limits:
          nvidia.com/gpu: 1
      image: "{{LEARNER_IMAGE}}"
      command:
        - /bin/bash
        - -c
        - "{{LEARNER_CMD}}"
---
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: worker-pods
  labels:
    tier: worker-set
spec:
  replicas: "{{NUM_WORKERS}}"
  selector:
    matchLabels:
      tier: worker-set
  template:
    metadata:
      labels:
        tier: worker-set
    spec:
      nodeSelector:
        nodetype: "{{PHORTX_NODE}}"
      containers:
        - name: worker-container
          tty: true
          stdin: true
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "{{GPU_ID}}"
            - name: CUDA_VISIBLE_DEVICES
              value: "{{GPU_ID}}"
          image: "{{WORKER_IMAGE}}"
          command:
            - "/bin/bash"
            - "-c"
            - "{{WORKER_CMD}}"
```

### Original Work

This is a revival of an original work that James Herman completed at Carnegie Mellon. For reference, the paper and experimental results of the distributed system are located at https://github.com/hermgerm29/distrib_l2r/assets/extending_l2r_with_distributed_deep_rl.pdf.
