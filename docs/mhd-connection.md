# MHD Connection

Topological Adam borrows structural language from the MHD closure project, but the relationship should be stated carefully.

## What Carries Over

- coupled auxiliary fields
- a regulated field-energy quantity
- a coupling signal tracked during updates
- interest in whether structured internal dynamics help control evolution

## What Does Not Carry Over

- this repository does not solve MHD equations
- `alpha`, `beta`, and `J_t` are not physical fields or currents here
- the optimizer experiments do not validate a plasma model

## Honest Cross-Repo Position

The MHD repository is the theory-facing source layer.
This repository is the applied optimizer branch that reuses some structural ideas.

That connection is worth documenting because it shaped the optimizer design. It is not a license to present optimizer behavior as physical evidence.
