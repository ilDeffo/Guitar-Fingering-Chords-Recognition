import os
import random

def remix():
    n = 0
    for f in os.listdir("."):
        new_name = f"Tastiera({n}).jpg"
        if f.endswith(".py"):
            continue
        os.rename(f, f"{new_name}")
        n += 1

remix()
