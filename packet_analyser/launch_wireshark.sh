#!/usr/bin/env bash
set -e

# Ajouter ton utilisateur au groupe Wireshark
sudo usermod -aG wireshark "$(id -un)"

# Utiliser le groupe immédiatement et lancer Wireshark
exec sg wireshark -c 'exec wireshark'