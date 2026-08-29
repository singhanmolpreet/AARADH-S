#!/bin/sh
# Brings up a virtual CAN interface for local testing.
# Linux / WSL2 only -- SocketCAN is a Linux kernel feature, not available
# on native Windows or macOS.
set -e

sudo modprobe vcan
sudo ip link add dev vcan0 type vcan 2>/dev/null || echo "vcan0 already exists"
sudo ip link set up vcan0

echo "vcan0 is up. Verify with: ip link show vcan0"
echo "Watch raw frames with:    candump vcan0   (install with: sudo apt install can-utils)"
