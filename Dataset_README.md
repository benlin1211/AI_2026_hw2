# Dataset

## Overview

Players communicate through a bulletin-board-style discussion, identify the werewolves, and vote to execute them.
If all werewolves are executed, the villagers win. If the number of villagers is reduced to the same as or fewer than the number of werewolves, the werewolves win. The player to be executed is determined by a vote among the surviving villagers.

Some villagers possess special abilities. By making effective use of these abilities, the village may be able to identify the werewolves more efficiently. Since these abilities can strongly influence the outcome of the game, players with special roles should try to survive as long as possible and lead the village to victory. Logs from games may be useful as reference.

The state of the village changes at each daily update. Before the update, players must set their vote target, divination target, and other required actions. Any player who does not post will die from sudden death.

---

## Glossary

* white = likely villager
* black = likely werewolf
* gray = unresolved
* GS = ranking from white to black among unresolved players
* CO = claim
* will = prewritten role reveal / instructions if attacked or killed
* confirmed town = role-confirmed non-wolf from village perspective
* panda = one white result and one black result on the same target

## How to Participate

New ID registration has been suspended.

---

## Game Flow

### Prologue

If at least 10 players have entered, the game proceeds to Day 1.

### Day 1

Each player learns their role and alignment.
Werewolves may use this phase to discuss and prepare their strategy.

### Day 2 and After

The villagers decide whom to execute through voting.
Use your role abilities and reasoning skills to pursue victory.

---

## Special Roles

### Villager

Has no special ability.

### Werewolf

Each night, may attack and kill one human player.
Werewolves can communicate with one another through a private channel that only werewolves can hear.
There are **2 werewolves** in villages of up to 12 players, and **3 werewolves** in villages of 13 or more players.

### Seer

Each night, may divine one player.
The Seer learns whether that player is a werewolf or a human.

### Medium

Can determine whether a player who died by execution or sudden death was a werewolf or a human.

### Madman

A human aligned with the werewolf side.
A werewolf victory is also a victory for the Madman.
This role is added in villages with **11 or more players**.
The Madman and the werewolves do **not** know each other’s identities.

### Hunter

Each night, may protect one player from a werewolf attack.
This role is added in villages with **11 or more players**.
The Hunter does **not** know whether the protection was successful.

## Predict

The dataset contains real player gameplay data for the first half of the scenario. You need to predict the role each character plays and the probability that they are werewolves.

id,index,character,role,wolf_score
1,01,Optimist Gerd,,