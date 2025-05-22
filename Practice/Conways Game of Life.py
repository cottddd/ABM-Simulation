import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid


class Cell(Agent):
    DEAD = 0
    ALIVE = 1

    def __init__(self, pos, model, init_state=DEAD):
        super().__init__(pos, model)
        self.x, self.y = pos
        self.state = init_state
        self._next_state = None

    @property
    def is_alive(self):
        return self.state == self.ALIVE

    def step(self):
        neighbors = self.model.grid.get_neighbors((self.x, self.y), moore=True, include_center=False)
        live_neighbors = sum(n.state for n in neighbors)

        if self.state == self.ALIVE:
            if live_neighbors < 2 or live_neighbors > 3:
                self._next_state = self.DEAD
            else:
                self._next_state = self.ALIVE
        else:
            if live_neighbors == 3:
                self._next_state = self.ALIVE
            else:
                self._next_state = self.DEAD

    def advance(self):
        self.state = self._next_state


class GameOfLife(Model):
    def __init__(self, width=20, height=20, initial_fraction_alive=0.2):
        super().__init__()
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = SimultaneousActivation(self)

        for cell_content, (x, y) in self.grid.coord_iter():
            cell = Cell((x, y), self)
            if self.random.random() < initial_fraction_alive:
                cell.state = cell.ALIVE
            self.grid.place_agent(cell, (x, y))
            self.schedule.add(cell)

    def step(self):
        self.schedule.step()


# Streamlit UI
st.title("Conway's Game of Life")

width = st.slider("Grid Width", 5, 50, 20)
height = st.slider("Grid Height", 5, 50, 20)
alive_fraction = st.slider("Initial Fraction Alive", 0.0, 1.0, 0.2)
steps = st.slider("Number of Steps", 1, 100, 10)

if st.button("Run Simulation"):
    model = GameOfLife(width=width, height=height, initial_fraction_alive=alive_fraction)
    for _ in range(steps):
        model.step()

    # 시각화용 2D 배열 만들기
    grid = np.zeros((height, width))
    for (x, y), cell in model.grid.coord_iter():
        grid[y, x] = cell.state

    st.subheader("Final State")
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="Greys", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)
