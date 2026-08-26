# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# Holds the Python process for tools/check_runtime.py when the host is Windows, which has
# no fcntl.  This image is NOT the submission image and is never measured: it only runs the
# official tool, which starts the containers it measures through the mounted Docker socket.
FROM docker:cli
RUN apk add --no-cache python3
