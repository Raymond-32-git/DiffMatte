import sys
import os
import importlib.util

_TARGET_REGISTRY = {}

class ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            v = self[name]
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                v = ConfigDict(v)
                self[name] = v
            return v
        # D2 configs might access missing keys before setting them
        # so we return a new ConfigDict to allow chained assignments
        # But for now, let's keep it simple.
        return super().__getattribute__(name)
        
    def __setattr__(self, name, value):
        self[name] = value

class LazyCallMock:
    def __init__(self, target):
        self._target_ = target
        
    def __call__(self, **kwargs):
        # We use ConfigDict instead of OmegaConf.create here to avoid baggage
        # and only use MockOmegaConf when the config explicitly asks for it.
        res = ConfigDict({"_target_": self._target_})
        if callable(self._target_):
            t_id = f"mock_target:{id(self._target_)}"
            _TARGET_REGISTRY[t_id] = self._target_
            res["_target_"] = t_id
            
        for k, v in kwargs.items():
            res[k] = v
        return res

    def __repr__(self):
        return f"LazyCall({self._target_})"

def LazyCall(target):
    return LazyCallMock(target)

class MockOmegaConf:
    @staticmethod
    def create(obj=None, **kwargs):
        return ConfigDict(obj) if obj is not None else ConfigDict()
    @staticmethod
    def to_container(obj, **kwargs):
        return dict(obj)
    @staticmethod
    def load(path):
        return ConfigDict()

MockOmegaConf.OmegaConf = MockOmegaConf

class LazyConfig:
    @staticmethod
    def load(path, name=None):
        import importlib.util
        path = os.path.abspath(path)
        
        # Determine package name and parent dir
        dir_name = os.path.dirname(path)
        pkg_name = os.path.basename(dir_name)
        mod_name = os.path.basename(path).replace(".py", "")
        
        if name is not None:
            full_mod_path = name
        else:
            full_mod_path = f"{pkg_name}.{mod_name}"
        
        # Ensure parent dir is in path
        parent_dir = os.path.dirname(dir_name)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
            
        # DYNAMIC OMEGACONF MOCKING
        # This prevents type errors when configs assign non-primitives to OmegaConf nodes
        import omegaconf
        real_omegaconf = sys.modules.get("omegaconf")
        sys.modules["omegaconf"] = MockOmegaConf
        
        try:
            spec = importlib.util.spec_from_file_location(full_mod_path, path)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = pkg_name
            
            # Register in sys.modules
            sys.modules[full_mod_path] = mod
            
            if pkg_name not in sys.modules:
                 pkg_spec = importlib.util.spec_from_loader(pkg_name, None, is_package=True)
                 pkg_mod = importlib.util.module_from_spec(pkg_spec)
                 pkg_mod.__path__ = [dir_name]
                 sys.modules[pkg_name] = pkg_mod

            # SOURCE TRANSFORMATION
            # This is the most reliable way to ensure all dict() calls use ConfigDict
            with open(path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Replace dict( with ConfigDict(
            # We use a simple replacement for configuration files
            source = source.replace("dict(", "ConfigDict(")
            
            # Also inject the class names into the scope
            mod.__dict__["ConfigDict"] = ConfigDict
            mod.__dict__["dict"] = ConfigDict
            
            # Execute the code in the module's dict
            exec(source, mod.__dict__)
            
            # Post-processing for literals like x = {...}
            for key, val in list(mod.__dict__.items()):
                if isinstance(val, dict) and not isinstance(val, ConfigDict) and not key.startswith("__"):
                    setattr(mod, key, ConfigDict(val))
                    
            return mod
        finally:
            if real_omegaconf:
                sys.modules["omegaconf"] = real_omegaconf
            else:
                del sys.modules["omegaconf"]

def instantiate(cfg):
    import functools
    if cfg is None:
        return None
        
    # Handle list of configs
    if isinstance(cfg, list):
        return [instantiate(v) for v in cfg]

    # Decide if we are looking at an instantiable object
    # Case A: Dictionary or OmegaConf DictConfig
    is_dict_like = isinstance(cfg, dict) or (hasattr(cfg, "get") and hasattr(cfg, "items"))
    
    if is_dict_like:
        target_name = cfg.get("_target_", None)
        if target_name is None:
             # Just a normal dict/container
             return {k: instantiate(v) for k, v in cfg.items()}
             
        params = {k: instantiate(v) for k, v in cfg.items() if k != "_target_"}
    # Case B: Mock objects or others with _target_ attribute
    elif hasattr(cfg, "_target_") and not isinstance(cfg, (str, bytes, functools.partial)):
        target_name = getattr(cfg, "_target_")
        # Extract params from attributes
        params = {}
        for k in dir(cfg):
            if k.startswith("_") or k in ("to_dict", "get", "items"): continue
            params[k] = instantiate(getattr(cfg, k))
    else:
        # Primitive or already instantiated (like a partial)
        return cfg

    # Resolve target
    target = None
    if isinstance(target_name, str):
        if target_name.startswith("mock_target:"):
            target = _TARGET_REGISTRY.get(target_name)
        else:
            try:
                parts = target_name.split(".")
                module_path = ".".join(parts[:-1])
                func_name = parts[-1]
                mod = importlib.import_module(module_path)
                target = getattr(mod, func_name)
            except Exception as e:
                print(f"[MockD2] Failed to import {target_name}: {e}")
                raise e
    else:
        target = target_name

    if target is None:
        return cfg # Not a target dict

    return target(**params)
