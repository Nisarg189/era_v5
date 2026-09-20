#!/bin/sh
# era_v5/assignment8 is the source of truth. This copies the app into the
# Netlify site folder so the two cannot drift. Run it after any edit.
set -e
here=$(cd "$(dirname "$0")" && pwd)
dst="$here/../../webapp/assignment8"
mkdir -p "$dst"
rsync -a --delete --exclude 'README.md' --exclude 'deploy.sh' "$here"/ "$dst"/
echo "copied to $dst"
