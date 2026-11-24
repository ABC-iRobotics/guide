#!/usr/bin/env bash
# Run the ```bash blocks of one README section verbatim: the image build IS the install test.
# usage: readme_steps.sh README.md <Section>     ("## <Section>" up to the next "## " heading)
# The GUIDE clone is skipped: the build context is the checkout being built.
set -eo pipefail
awk -v h="## $2" '
  $0 == h                {on = 1; next}
  on && !code && /^## /  {on = 0}
  on && /^```bash/       {code = 1; next}
  code && /^```/         {code = 0; next}
  on && code' "$1" | grep -v 'ABC-iRobotics/guide.git' > /tmp/readme_steps.sh
cat /tmp/readme_steps.sh
[ -n "${DRY_RUN:-}" ] && exit 0
bash -exo pipefail /tmp/readme_steps.sh   # no -u: ROS setup scripts read unset variables
