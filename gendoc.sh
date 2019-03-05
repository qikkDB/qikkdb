#!/bin/bash
export GIT_HEAD_HASH="$(git rev-parse HEAD | cut -c1-8 ; git diff-index --quiet HEAD || echo '(with uncommitted changes)')"
exec doxygen ./doxygen.cfg

