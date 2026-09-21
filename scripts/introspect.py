import inspect, laya
from laya import Router
print("laya exports:", [n for n in dir(laya) if not n.startswith("_")])
print("\nload:", inspect.signature(laya.load))
print("\nRouter.__init__:", inspect.signature(Router.__init__))
print("Router methods:", [n for n in dir(Router) if not n.startswith("_")])
agent_cls = None
import laya.agent as A
for n in dir(A):
    o = getattr(A, n)
    if inspect.isclass(o) and hasattr(o, "predict"):
        agent_cls = o
        print(f"\nAgent class {n}.__init__:", inspect.signature(o.__init__))
        print(f"{n}.predict:", inspect.signature(o.predict))
        print(f"{n} methods:", [m for m in dir(o) if not m.startswith("_")])
