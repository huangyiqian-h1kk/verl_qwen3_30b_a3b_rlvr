# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extracted from verl commit 62ac6d6e9fccc09ab6af2171217cee6e290b799b.
# CPU tests replace Ray and training dependencies, not these functions.

def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """

    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        except Exception:
            # if it is a property, it will fail because Class doesn't have instance property
            continue

        if hasattr(method, MAGIC_ATTR):

            def generate_function(name, key=key):
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                async def async_func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return await getattr(self.worker_dict[key], name)(*args, **kwargs)

                wrapper = async_func if inspect.iscoroutinefunction(method) else func  # noqa: B023

                return wrapper

            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            attrs = getattr(method, MAGIC_ATTR)
            setattr(func, MAGIC_ATTR, attrs)
            try:
                # bind direct rollout method to class without prefix
                if attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD and "rollout" in key:
                    assert not hasattr(cls, method_name), (
                        f"conflict direct rollout method {method_name} with role {key}"
                    )
                    setattr(cls, method_name, func)
                    print(f"bind role {key} method {method_name} to class {cls}")
                else:
                    method_name_with_prefix = key + "_" + method_name
                    setattr(cls, method_name_with_prefix, func)
            except Exception as e:
                raise ValueError(f"Fail to set method_name {method_name}") from e


def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


def _determine_fsdp_megatron_base_class(mros: list):
    """
    - megatron: base class should be MegatronWorker
    - fsdp: base class should be Worker
    """
    for cls in mros[0]:
        if cls.__name__ == "MegatronWorker":
            return cls
        if cls.__name__ == "Worker":
            return cls
    raise ValueError(f"Cannot determine base class for {mros}")


def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = _determine_fsdp_megatron_base_class(
        [cls.cls.__ray_actor_class__.__mro__ for cls in class_dict.values()]
    )
    assert issubclass(worker_cls, Worker), f"worker_cls {worker_cls} should be a subclass of Worker"
    print(f"colocated worker base class {worker_cls}")

    for key, cls in class_dict.items():
        cls_dict[key] = cls.cls
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                # in worker class, e.g. <verl.single_controller.base.worker.Worker>
                # when DISABLE_WORKER_INIT == 1 it will return immediately
                with temp_env_var("DISABLE_WORKER_INIT", "1"):
                    self.worker_dict[key] = user_defined_cls(
                        *init_args_dict[key].get("args", ()), **init_args_dict[key].get("kwargs", {})
                    )

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(cls=remote_cls)
    return remote_cls


