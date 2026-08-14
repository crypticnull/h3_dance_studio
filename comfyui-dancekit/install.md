# Installing comfyui-dancekit

Two things need to be in place: this node pack in `custom_nodes/`, and the `dancekit`
package importable by ComfyUI's Python.

## 1. The node pack

```bash
cd ComfyUI/custom_nodes
git clone <this repo> comfyui-dancekit          # or copy the folder
```

## 2. dancekit itself

Pick ONE of these — the nodes look for dancekit in this order:

### a. pip install (recommended)

Into the *same* Python environment ComfyUI runs in:

```bash
# portable ComfyUI on Windows:
python_embeded\python.exe -m pip install -e C:\path\to\dancekit
# venv / system install:
pip install -e /path/to/dancekit
```

(`/path/to/dancekit` is the repo folder containing `README.md` and the inner
`dancekit/` package.)

### b. vendor / symlink next to the node pack

No pip needed. Put the inner `dancekit` *package folder* (the one containing
`skeleton.py`, `compose.py`, ...) in any of:

```
custom_nodes/comfyui-dancekit/dancekit/       # vendored inside the pack
custom_nodes/dancekit/                        # bare package as a sibling
custom_nodes/dancekit/dancekit/               # full repo checkout as a sibling
```

Symlinks work too:

```bash
ln -s /path/to/dancekit/dancekit ComfyUI/custom_nodes/comfyui-dancekit/dancekit
```

## 3. Dependencies

```bash
pip install -r custom_nodes/comfyui-dancekit/requirements.txt
```

torch is not listed on purpose — ComfyUI already has it.

Optional, only if you want **Harvest** to run pose detection on *video* files
(harvesting ComfyUI pose JSONs needs nothing extra):

```bash
pip install rtmlib onnxruntime-gpu
```

## 4. Check

Restart ComfyUI. The nodes appear under the **dancekit** category. If loading fails
with "dancekit is not importable", step 2 didn't land in the environment ComfyUI
actually uses — the error message lists the exact paths that were searched.

You can also sanity-check outside ComfyUI:

```bash
python custom_nodes/comfyui-dancekit/selftest.py
```
