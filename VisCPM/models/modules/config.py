import json
import os
import copy
from typing import Any, Dict, Union


class Config(object):
    """model configuration"""

    def __init__(self):
        super().__init__()

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **args):
        config_dict = cls._dict_from_json_file(json_file, **args)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike], **args):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        res = json.loads(text)
        for key in args:
            res[key] = args[key]
        return res

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self) -> str:
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output
