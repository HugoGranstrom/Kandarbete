import torch.nn as nn
import torch.nn.functional as F
import os
import torch

def saveNetGAN(filename, generator, optimizer_gen, discriminator, optimizer_disc, iterations, train_loss, val_loss):
  torch.save({
      "gen": generator.state_dict(),
      "optimizer_gen": optimizer_gen.state_dict(),
      "disc": discriminator.state_dict(),
      "optimizer_disc": optimizer_disc.state_dict(),
      "iteration": iterations,
      "loss": train_loss,
      "val_loss": val_loss
  }, filename)

def loadNetGAN(filename, generator, optimizer_gen, discriminator, optimizer_disc, device):
  try:
    checkpoint = torch.load(filename, map_location=device)
    generator.load_state_dict(checkpoint["gen"])
    optimizer_gen.load_state_dict(checkpoint["optimizer_gen"])
    discriminator.load_state_dict(checkpoint["disc"])
    optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
    iteration = checkpoint["iteration"]
    train_loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    print(f"Net loaded from memory! Starting on iteration {iteration[-1]+1} with train-loss {train_loss[-1]}")
    return iteration, train_loss, val_loss
  except (OSError, FileNotFoundError):
    print(f"Couldn't find {filename}, creating new net!")
    return [], [], []




def saveNet(filename, net, optimizer, iterations, train_loss, val_loss):
  torch.save({
      "gen": net.state_dict(),
      "optimizer": optimizer.state_dict(),
      "iteration": iterations,
      "loss": train_loss,
      "val_loss": val_loss
  }, filename)

def loadNet(filename, net, optimizer, device):
  try:
    checkpoint = torch.load(filename, map_location=device)
    net.load_state_dict(checkpoint["gen"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    iteration = checkpoint["iteration"]
    train_loss = checkpoint["loss"]
    val_loss = checkpoint["val_loss"]
    print(f"Net loaded from memory! Starting on iteration {iteration[-1]+1} with train-loss {train_loss[-1]}")
    return iteration, train_loss, val_loss
  except (OSError, FileNotFoundError):
    print(f"Couldn't find {filename}, creating new net!")
    return [], [], []

def loadNetEval(filename, net, device):
  try:
    checkpoint = torch.load(filename, map_location=device)
    net.load_state_dict(checkpoint["gen"])
    net.eval()
    print(filename, "successfully loaded in eval mode")
  except (OSError, FileNotFoundError):
    print(f"Couldn't find {filename}")