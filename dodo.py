"""Project builder."""

import pathlib
from glob import glob

import pygraphviz
from import_deps import ModuleSet

DOIT_CONFIG = {
    "default_tasks": ["imports", "dot", "draw", "codestyle"],
}

node_color = "#d0a9d0"
output_name = "dependencies"
base_path = pathlib.Path("ipcv_tools")
modules = [
    path for path in base_path.glob("**/*.py") if pathlib.Path(path).stem != "__init__"
]
PKG_MODULES = ModuleSet(modules)


def get_imports(pkg_modules, module_path):
    module = pkg_modules.by_path[module_path]
    imports = pkg_modules.get_imports(module, return_fqn=True)
    return {"modules": list(sorted(imports))}


def task_imports():
    """Find imports from a python module."""
    for name, module in PKG_MODULES.by_name.items():
        yield {
            "name": name,
            "file_dep": [module.path],
            "actions": [(get_imports, (PKG_MODULES, module.path))],
        }


def print_imports(modules):
    print("\n".join(modules))


def task_print():
    """Print on stdout list of direct module imports."""
    for name, module in PKG_MODULES.by_name.items():
        yield {
            "name": name,
            "actions": [print_imports],
            "getargs": {"modules": ("imports:{}".format(name), "modules")},
            "uptodate": [False],
            "verbosity": 2,
        }


def module_to_dot(imports, targets):
    graph = pygraphviz.AGraph(strict=False, directed=True)
    graph.node_attr["fillcolor"] = node_color
    graph.node_attr["style"] = "filled"
    for source, sinks in imports.items():
        for sink in sinks:
            graph.add_edge(source, sink)
    graph.write(targets[0])


def task_dot():
    """Generate a graphviz's dot graph from module imports."""
    return {
        "targets": [f"{output_name}.dot"],
        "actions": [module_to_dot],
        "getargs": {"imports": ("imports", "modules")},
        "clean": True,
    }


def task_draw():
    """Generate image from a dot file."""
    return {
        "file_dep": [f"{output_name}.dot"],
        "targets": [f"{output_name}.png"],
        "actions": ["dot -Tpng -Gdpi=70 %(dependencies)s -o %(targets)s"],
        "clean": True,
    }


def task_codestyle():
    """Ensure proper codestyle."""
    files = []
    for folder in ["examples", "ipcv_tools"]:
        files.extend(glob(f"{folder}/*.py"))
    return {
        "file_dep": files,
        "actions": ["isort -q .", "black -q ."],
    }


if __name__ == "__main__":
    for subtask in task_imports():
        print(subtask)
    print(task_dot())
    print(task_draw())
