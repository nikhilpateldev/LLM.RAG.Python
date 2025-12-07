
from typing import Tuple, List
import importlib, traceback

def _import(name):
    try:
        return importlib.import_module(name)
    except:
        return None

base_rag = _import("base_rag")
mod_b = _import("approach_b_conditional")
mod_c = _import("approach_c_hybrid")
mod_d = _import("approach_d_router")
mod_e = _import("approach_e_multiquery")

class UnifiedAgent:
    def __init__(self):
        print("UnifiedAgent ready. Approaches:")
        print("B Conditional:", bool(mod_b))
        print("C Hybrid:", bool(mod_c))
        print("D Router:", bool(mod_d))
        print("E MultiQuery:", bool(mod_e))

    def answer_conditional(self, q):
        return mod_b.answer_question_conditional(q)

    def answer_hybrid(self, q):
        return mod_c.answer_hybrid(q)

    def answer_router(self, q):
        return mod_d.answer_router(q)

    def answer_multiquery(self, q):
        return mod_e.answer_multi(q)

    def run_mode(self, mode, q):
        mode = mode.lower()
        if mode=="conditional":
            ans,src = self.answer_conditional(q)
        elif mode=="hybrid":
            ans,src = self.answer_hybrid(q)
        elif mode=="router":
            ans,src = self.answer_router(q)
        elif mode=="multi":
            ans,src = self.answer_multiquery(q)
        else:
            raise ValueError("Unknown mode")
        return {"mode":mode,"question":q,"answer":ans,"sources":src}
