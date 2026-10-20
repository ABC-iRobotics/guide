import sys

from guide_ex.recipe.catalogue import write

if sys.argv[1:] != ["catalogue"]:
    sys.exit("usage: python -m guide_ex.recipe catalogue")
for path in write():
    print(f"wrote {path}")
