#!/usr/bin/env bash
# Pulls a curated set of real Kubernetes docs (official kubernetes/website
# repo — the source of kubernetes.io) into ./corpus for ingest_k8s_docs.py.
set -e
rm -rf k8s-website corpus
mkdir -p corpus

git clone --depth 1 --filter=blob:none --sparse https://github.com/kubernetes/website.git k8s-website
cd k8s-website
git sparse-checkout set --skip-checks \
  content/en/docs/concepts/workloads/pods/_index.md \
  content/en/docs/concepts/workloads/pods/pod-lifecycle.md \
  content/en/docs/concepts/workloads/controllers/deployment.md \
  content/en/docs/concepts/workloads/controllers/statefulset.md \
  content/en/docs/concepts/workloads/controllers/job.md \
  content/en/docs/concepts/services-networking/service.md \
  content/en/docs/concepts/services-networking/network-policies.md \
  content/en/docs/concepts/services-networking/ingress.md \
  content/en/docs/concepts/configuration/configmap.md \
  content/en/docs/concepts/configuration/secret.md \
  content/en/docs/concepts/storage/volumes.md \
  content/en/docs/concepts/storage/persistent-volumes.md \
  content/en/docs/concepts/security/rbac-good-practices.md \
  content/en/docs/concepts/security/pod-security-standards.md

for f in $(git sparse-checkout list >/dev/null 2>&1; find content -name "*.md"); do
  cp "$f" "../corpus/$(basename "$f")"
done
cd ..
rm -rf k8s-website
echo "corpus ready:"
ls corpus/
