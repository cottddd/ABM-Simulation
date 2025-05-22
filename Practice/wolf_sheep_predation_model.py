import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

class GrassPatch(Agent):
    def __init__(self, pos, model, fully_grown=True, regrowth_time=30):
        super().__init__(pos, model)
        self.pos = pos
        self.fully_grown = fully_grown
        self.countdown = regrowth_time

    def step(self):
        if not self.fully_grown:
            self.countdown -= 1
            if self.countdown <= 0:
                self.fully_grown = True
                self.countdown = self.model.grass_regrowth_time

class Sheep(Agent):
    def __init__(self, unique_id, model, energy=4):
        super().__init__(unique_id, model)
        self.energy = energy

    def step(self):
        self.move()
        self.energy -= 1

        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for obj in cellmates:
            if isinstance(obj, GrassPatch) and obj.fully_grown:
                self.energy += self.model.sheep_gain_from_food
                obj.fully_grown = False
                break

        if self.energy > 0 and self.random.random() < self.model.sheep_reproduce:
            self.model.new_agents.append(
                Sheep(self.model.next_id(), self.model, self.energy / 2)
            )
            self.energy /= 2
        elif self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

class Wolf(Agent):
    def __init__(self, unique_id, model, energy=8):
        super().__init__(unique_id, model)
        self.energy = energy

    def step(self):
        self.move()
        self.energy -= 1

        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        sheep = [obj for obj in cellmates if isinstance(obj, Sheep)]
        if sheep:
            target = self.random.choice(sheep)
            self.energy += self.model.wolf_gain_from_food
            self.model.grid.remove_agent(target)
            self.model.schedule.remove(target)

        if self.energy > 0 and self.random.random() < self.model.wolf_reproduce:
            self.model.new_agents.append(
                Wolf(self.model.next_id(), self.model, self.energy / 2)
            )
            self.energy /= 2
        elif self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

class WolfSheepModel(Model):
    def __init__(self, width, height, init_sheep, init_wolves, sheep_gain_from_food, wolf_gain_from_food,
                 sheep_reproduce, wolf_reproduce, grass_regrowth_time, enable_grass=True):
        super().__init__()
        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)
        self.enable_grass = enable_grass
        self.sheep_gain_from_food = sheep_gain_from_food
        self.wolf_gain_from_food = wolf_gain_from_food
        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.grass_regrowth_time = grass_regrowth_time
        self.new_agents = []

        if self.enable_grass:
            for _, (x, y) in self.grid.coord_iter():
                patch = GrassPatch((x, y), self, fully_grown=self.random.choice([True, False]))
                self.grid.place_agent(patch, (x, y))
                self.schedule.add(patch)

        for _ in range(init_sheep):
            sheep = Sheep(self.next_id(), self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(sheep, (x, y))
            self.schedule.add(sheep)

        for _ in range(init_wolves):
            wolf = Wolf(self.next_id(), self)
            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(wolf, (x, y))
            self.schedule.add(wolf)

        self.datacollector = DataCollector(
            model_reporters={
                "Sheep": lambda m: sum(isinstance(a, Sheep) for a in m.schedule.agents),
                "Wolves": lambda m: sum(isinstance(a, Wolf) for a in m.schedule.agents),
                "Grass": lambda m: sum(
                    isinstance(a, GrassPatch) and a.fully_grown for a in m.schedule.agents
                ),
            }
        )
        self.datacollector.collect(self)

    def step(self):
        self.schedule.step()
        for agent in self.new_agents:
            self.grid.place_agent(agent, agent.pos)
            self.schedule.add(agent)
        self.new_agents = []
        self.datacollector.collect(self)

# Streamlit App
st.title("Wolf-Sheep Predation Model")

cols = st.columns(2)
with cols[0]:
    width = st.slider("Grid Width", 10, 50, 20)
    init_sheep = st.slider("Initial Sheep", 10, 300, 100)
    sheep_gain = st.slider("Sheep Gain from Food", 1, 10, 4)
    sheep_reproduce = st.slider("Sheep Reproduction Rate", 0.01, 1.0, 0.04, 0.01)
    grass = st.checkbox("Enable Grass", True)

with cols[1]:
    height = st.slider("Grid Height", 10, 50, 20)
    init_wolves = st.slider("Initial Wolves", 10, 100, 50)
    wolf_gain = st.slider("Wolf Gain from Food", 1, 30, 20)
    wolf_reproduce = st.slider("Wolf Reproduction Rate", 0.01, 1.0, 0.05, 0.01)
    regrowth_time = st.slider("Grass Regrowth Time", 1, 100, 30)

steps = st.slider("Simulation Steps", 1, 200, 50)

if st.button("Run Simulation"):
    model = WolfSheepModel(
        width, height, init_sheep, init_wolves,
        sheep_gain, wolf_gain,
        sheep_reproduce, wolf_reproduce,
        regrowth_time, enable_grass=grass
    )

    for _ in range(steps):
        model.step()

    df = model.datacollector.get_model_vars_dataframe()

    st.subheader("Population Over Time")
    st.line_chart(df)

    st.subheader("Final State Grid")
    grid_img = np.zeros((height, width, 3))

    for _, (x, y) in model.grid.coord_iter():
        cell_agents = model.grid.get_cell_list_contents([(x, y)])
        color = [1.0, 1.0, 1.0]

        for a in cell_agents:
            if isinstance(a, GrassPatch):
                color = [0.0, 0.6, 0.0] if a.fully_grown else [0.5, 0.3, 0.1]
            elif isinstance(a, Sheep):
                color = [0.5, 0.9, 1.0]
            elif isinstance(a, Wolf):
                color = [1.0, 0.2, 0.2]

        grid_img[y, x] = color

    fig, ax = plt.subplots()
    ax.imshow(grid_img)
    ax.set_xticks([]), ax.set_yticks([])
    st.pyplot(fig)
