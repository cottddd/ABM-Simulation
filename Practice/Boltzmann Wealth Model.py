import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class MoneyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        cellmates = [a for a in cellmates if a != self]
        if cellmates:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1

    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()


class BoltzmannWealth(Model):
    def __init__(self, n=100, width=10, height=10, seed=None):
        super().__init__(seed=seed)
        self.num_agents = n
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        self.datacollector = DataCollector(
            model_reporters={"Gini": self.compute_gini},
            agent_reporters={"Wealth": "wealth"},
        )

        for i in range(self.num_agents):
            agent = MoneyAgent(i, self)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def compute_gini(self):
        agent_wealths = [agent.wealth for agent in self.schedule.agents]
        if sum(agent_wealths) == 0:
            return 0
        x = sorted(agent_wealths)
        n = self.num_agents
        b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
        return 1 + (1 / n) - 2 * b


st.title("Boltzmann Wealth Model Simulation")

num_agents = st.slider("Number of Agents", 10, 100, 50)
width = st.slider("Grid Width", 5, 20, 10)
height = st.slider("Grid Height", 5, 20, 10)
num_steps = st.slider("Number of Steps", 1, 200, 100)
seed = st.number_input("Random Seed", value=42)

if st.button("Run Simulation"):
    model = BoltzmannWealth(n=num_agents, width=width, height=height, seed=int(seed))
    for _ in range(num_steps):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    st.subheader("Gini Coefficient Over Time")
    fig, ax = plt.subplots()
    ax.plot(model_df["Gini"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Gini Coefficient")
    st.pyplot(fig)

    st.subheader("Final Agent Wealth Distribution")
    final_wealth = agent_df.loc[agent_df.index.get_level_values(0) == num_steps - 1]
    st.bar_chart(final_wealth["Wealth"].value_counts().sort_index())
