import sys

if sys.argv[1:] == ["catalogue"]:
    from guide_ex.recipe.catalogue import write

    for path in write():
        print(f"wrote {path}")
elif sys.argv[1:] == ["view"]:
    from guide_ex.recipe.viewer import main

    main()
else:
    sys.exit("usage: python -m guide_ex.recipe catalogue|view")
