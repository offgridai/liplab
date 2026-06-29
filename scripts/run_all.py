import pathlib, subprocess, sys
root = pathlib.Path(__file__).resolve().parents[1]
bin_name = 'liplab_runner.exe' if sys.platform.startswith('win') else 'liplab_runner'
candidates = [root/'build'/bin_name, root/'build'/'Release'/bin_name, root/'build'/'Debug'/bin_name]
exe = next((p for p in candidates if p.exists()), None)
if not exe:
    raise SystemExit('liplab_runner not built; run cmake first')
raise SystemExit(subprocess.call([str(exe), str(root)]))
