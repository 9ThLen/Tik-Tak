"""The two registered S1 training arms.

A two-element tuple of strings lives in its own module because it used to live
in `trainer`, which imports torch. `summarise` imports the constant, and
`c1_summarise` imports `bootstrap` from `summarise`, so naming the arms pulled
torch into the C1 verdict logic and its tests could not be collected in an
environment without it. Nothing here should ever grow a heavy dependency.
"""

ARMS = ("A3_reset", "A3_stateful")
