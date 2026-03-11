---
layout: post
title: 'Programmer in Imprisonment Problem'
date: 2026-03-11 10:44
description: "A thought for a logical problem"
tags: [Programming Logic Distributed_System]
categories: Programming
tabs: true
---

Background: 100 people are imprisoned in a dungeon. You are one of them and you are also a programmer, you have one chance to talk to everyone in less than 3 min. Then every one of you will have a prob. to be released out to a large room with a bulb can be turned on and off. You need to answer if everyone has been released for at least once. If correct, everyone will be set free instantly, if not, everyone will be killed. Each person can't communication to each other except for the first 3 min.

Question: How to determine whether if all the people have been released at least once?

Strategy 1: For each person, if it is your first time to go out when you find the bulb is off, turn it on. When the programmer releases, he will turn it off and set a counter plus one. Then idealy if the programmer counts to 99, it means everyone has been released for once, Horay!

Problem: When the bulb is initially turned on, the programmer will accecidently count one more person.

How to solve: For each person, if it is your first two time to go out when you find the bulb is off, turn it on. When the programmer releases, he will turn it off and set a counter plus one. Then when the counter reaches 198, no matter the bulb was turn on or off initially, everyone has been released at least once(99*2 or 98*2+1+1,1 for 1 person came out and 1 for bulb was on at first)
