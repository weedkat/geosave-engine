import albumentations as A

class TransformsCompose:
    """
    Compose a list of transformations specified in the config.
    """
    def __init__(self, cfg=None, input_size=None):
        
        if not isinstance(cfg, list):
            raise ValueError("Expected a list of transform specifications")
        
        self.input_size = input_size

        transforms = [self.build_transforms(spec) for spec in cfg]
        
        self.transform = A.Compose(transforms)
     
    def __call__(self, **kwargs):
        return self.transform(**kwargs)

    def build_transforms(self, spec):
        name = spec['name']
        args = spec.get('kwargs', {}).copy()
        cls = getattr(A, name)
        
        if name in ("OneOf", "SomeOf", "Compose"):
            nested_spec = args.pop('transforms', [])
            transforms = [self.build_transforms(t) for t in nested_spec]
            return cls(transforms, **args)

        if name in ("RandomResizedCrop", "RandomCrop", "CenterCrop", "Resize"):
            if self.input_size is not None:
                args['size'] = [self.input_size, self.input_size]
            elif 'size' not in args:
                raise ValueError(f"{name} requires 'size' in kwargs or input_size passed to TransformsCompose")
            
        return cls(**args)

    def __add__(self, other):
        if not isinstance(other, TransformsCompose):
            raise ValueError("Can only add another TransformsCompose instance")
        
        new = TransformsCompose()
        new.transform = A.Compose(self.transform.transforms + other.transform.transforms)
        return new