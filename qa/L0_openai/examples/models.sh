#!/bin/bash
curl -s http://localhost:8000/v1/models \
  -H "Content-Type: application/json" | jq
