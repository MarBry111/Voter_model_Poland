# Voter_model_Poland
Voter model with extra features - the goal is to simulate behaviour in Poland since 2005.

## Step 1: <br>
Project to model the voting situation in Poland for Europarlament Election (2004-2019). It is really easy model, with preferences of voting for one of two parties for each district. Finally we ended with 26 parameters, for 2 for each node, which are constant in time. The voer model used in this project is modified. In every time step the preferences of each node changed, depending on the neighbours' support for each paty. <br>
There was 5% chance to change the preferences on bit higher or lower for each party at each time step.
#### Conclusions: <br>
1. The outcome of rundom support for each party is that, the 40% of situations ended with the domination of party A, the next 40% of the domination of party B and 20% of the results where no party dominates. The interesting fact is that the 43% of this 20% is the situation where Poland is divided on the West side and East side (just like in real life).
<p float="left">
  <img src="random-poland/14_4remisy.png" width="200"/>
  <img src="random-poland/14_7remisy.png" width="200"/>
  <img src="random-poland/7_2remisy.png"  width="200"/>
  <img src="random-poland/7_3remisy.png"  width="200"/>
</p>
2. Using genetic algorithm there was find the best parameters to model the situation in last 15 years. It gave about 90% of accuracy when the two parameters for district Łódzie was chenaged by had. The preferences where summaring for 4 years and the voting change the state in 5th year to wich was more possible after this time.<br>  <br>
3. For th best results it should be used more precise map with more nodes (not only 13) and probably the smaller time stap than 1 year. Adding more parameters could be also helpfull, and watching how they change and if they correspond to the actual value of some parameters like the average income, education or age.

## Step 2: <br>
The next step is to check which of variables describing each district are most influencive for th outcome of election. ,br.
First fo all I decidet ot use the data only for parlamentary elections due to fact that people treat it like the most important [[1]](#1) [[2]](#2). Then based on the number of politics from each party in parlament [[3]](#3) I decided to focuse on elections from 2005 - 2019 beacause of fact the simplyfing model to two parties only, one Civic Platform (PO) and second Law and Juistice (PiS).


## References
<a id="1">[1]</a> Waldemar Wojtasik (2010). Drugorzędność wyborów samorządowych w teorii i badaniach empirycznych. <br>
<a id="2">[2]</a> Michael Marsh (1998). Testing the Second-Order Election Modelafter Four European Elections. <br>
<a id="3">[3]</a> https://pl.wikipedia.org/wiki/Wybory_w_Polsce#/media/Plik:Procentowe_wyniki_wybor%C3%B3w_do_Sejmu.png <br>

![Alt text](powiaty.png?raw=true "powiaty")
