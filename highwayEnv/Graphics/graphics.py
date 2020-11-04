from __future__ import division, print_function, absolute_import

import os
import numpy as np
import pygame

from highwayEnv.road.lane import LineType
from highwayEnv.vehicles.control import Vehicle,AgentVehicle,simpleVehicle, LinearVehicle



class EnvViewer(object):
    """
        A viewer to render a highway driving environment.
    """
    SAVE_IMAGES = False

    def __init__(self, env, offscreen=False):
        self.env = env
        self.offscreen = offscreen

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.env.config["screen_width"], self.env.config["screen_height"])

        # A display is not mandatory to draw things. Ignoring the display.set_mode()
        # instruction allows the drawing to be done on surfaces without
        # handling a screen display, useful for e.g. cloud computing
        if not self.offscreen:
            self.screen = pygame.display.set_mode([self.env.config["screen_width"], self.env.config["screen_height"]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = env.config.get("scaling", self.sim_surface.INITIAL_SCALING)
        self.sim_surface.centering_position = env.config.get("centering_position", self.sim_surface.INITIAL_CENTERING)
        self.clock = pygame.time.Clock()

        self.enabled = True
        if "SDL_VIDEODRIVER" in os.environ and os.environ["SDL_VIDEODRIVER"] == "dummy":
            self.enabled = False

        self.agent_display = None
        self.agent_surface = None
        self.vehicle_trajectory = None
        self.frame = 0

    def set_agent_display(self, agent_display):
        """
            Set a display callback provided by an agent, so that they can render their behaviour on a dedicated
            agent surface, or even on the simulation surface.
        :param agent_display: a callback provided by the agent to display on surfaces
        """
        if self.agent_display is None:
            if self.env.config["screen_width"] > self.env.config["screen_height"]:
                self.screen = pygame.display.set_mode((self.env.config["screen_width"],
                                                       2 * self.env.config["screen_height"]))
            else:
                self.screen = pygame.display.set_mode((2 * self.env.config["screen_width"],
                                                       self.env.config["screen_height"]))
            self.agent_surface = pygame.Surface((self.env.config["screen_width"], self.env.config["screen_height"]))
        self.agent_display = agent_display

        """ TODO: Check if this is necessary """
    #def set_agent_action_sequence(self, actions):
        """
            Set the sequence of actions chosen by the agent, so that it can be displayed
        :param actions: list of action, following the env's action space specification
        """
     #   if hasattr(self.env.action_space, 'n'):
     #       actions = [self.env.ACTIONS[a] for a in actions]
     #   self.vehicle_trajectory = self.env.vehicle.predict_trajectory(actions,
     #                                                                 1 / self.env.config["policy_frequency"],
     #                                                                 1 / 3 / self.env.config["policy_frequency"],
     #                                                                 1 / self.env.SIMULATION_FREQUENCY)

    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicle.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.ego_vehicles:
                VehicleGraphics.handle_event(self.env.ego_vehicles, event)

    def display(self):
        """
            Display the road and vehicles on a pygame window.
        """
        if not self.enabled:
            return

        """ 
            Line to display a static window instead of a moving one, if you want a moving one following the ego_vehicle, uncomment the first one
        """
        #self.sim_surface.move_display_window_to(self.window_position())
        self.sim_surface.move_display_window_to()
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)
        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen)

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if self.env.config["screen_width"] > self.env.config["screen_height"]:
                self.screen.blit(self.agent_surface, (0, self.env.config["screen_height"]))
            else:
                self.screen.blit(self.agent_surface, (self.env.config["screen_width"], 0))

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            self.clock.tick(self.env.SIMULATION_FREQUENCY)
            pygame.display.flip()

        if self.SAVE_IMAGES:
            pygame.image.save(self.screen, "highway-env_{}.png".format(self.frame))
            self.frame += 1

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.sim_surface)
        return np.moveaxis(data, 0, 1)

    def window_position(self):
        """
        :return: the world position of the center of the displayed window.
        """
        if self.env.vehicle:
            return self.env.vehicle.position
        else:
            return np.array([0, 0])

    def close(self):
        """
            Close the pygame window.
        """
        pygame.quit()

class LaneGraphics(object):
    """
        A visualization of a lane.
    """
    STRIPE_SPACING = 5
    """ Offset between stripes [m]"""

    STRIPE_LENGTH = 3
    """ Length of a stripe [m]"""

    STRIPE_WIDTH = 0.3
    """ Width of a stripe [m]"""

    @classmethod
    def display(cls, lane, surface):
        """
            Display a lane on a surface.

        :param lane: the lane to be displayed
        :param surface: the pygame surface
        """
        stripes_count = int(2 * (surface.get_height() + surface.get_width()) / (cls.STRIPE_SPACING * surface.scaling))
        s_origin, _ = lane.local_coordinates(surface.origin)
        s0 = (int(s_origin) // cls.STRIPE_SPACING - stripes_count // 2) * cls.STRIPE_SPACING
        for side in range(2):
            if lane.line_types[side] == LineType.STRIPED:
                cls.striped_line(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS:
                cls.continuous_curve(lane, surface, stripes_count, s0, side)
            elif lane.line_types[side] == LineType.CONTINUOUS_LINE:
                cls.continuous_line(lane, surface, stripes_count, s0, side)

    @classmethod
    def striped_line(cls, lane, surface, stripes_count, s0, side):
        """
            Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param s0: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_LENGTH
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_curve(cls, lane, surface, stripes_count, s0, side):
        """
            Draw a striped line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes to draw
        :param s0: the longitudinal position of the first stripe [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING
        ends = s0 + np.arange(stripes_count) * cls.STRIPE_SPACING + cls.STRIPE_SPACING
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def continuous_line(cls, lane, surface, stripes_count, s0, side):
        """
            Draw a continuous line on one side of a lane, on a surface.

        :param lane: the lane
        :param surface: the pygame surface
        :param stripes_count: the number of stripes that would be drawn if the line was striped
        :param s0: the longitudinal position of the start of the line [m]
        :param side: which side of the road to draw [0:left, 1:right]
        """
        starts = [s0 + 0 * cls.STRIPE_SPACING]
        ends = [s0 + stripes_count * cls.STRIPE_SPACING + cls.STRIPE_LENGTH]
        lats = [(side - 0.5) * lane.width_at(s) for s in starts]
        cls.draw_stripes(lane, surface, starts, ends, lats)

    @classmethod
    def draw_stripes(cls, lane, surface, starts, ends, lats):
        """
            Draw a set of stripes along a lane.

        :param lane: the lane
        :param surface: the surface to draw on
        :param starts: a list of starting longitudinal positions for each stripe [m]
        :param ends:  a list of ending longitudinal positions for each stripe [m]
        :param lats: a list of lateral positions for each stripe [m]
        """
        starts = np.clip(starts, 0, lane.length)
        ends = np.clip(ends, 0, lane.length)
        for k in range(len(starts)):
            if abs(starts[k] - ends[k]) > 0.5 * cls.STRIPE_LENGTH:
                pygame.draw.line(surface, surface.WHITE,
                                 (surface.vec2pix(lane.position(starts[k], lats[k]))),
                                 (surface.vec2pix(lane.position(ends[k], lats[k]))),
                                 max(surface.pix(cls.STRIPE_WIDTH), 1))

class RoadGraphics(object):
    """
        A visualization of a road lanes and vehicles.
    """
    @classmethod
    def display(cls, road, surface):
        """
            Display the road lanes on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        """
        surface.fill(surface.GREY)
        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for l in road.network.graph[_from][_to]:
                    LaneGraphics.display(l, surface)

    @classmethod
    def display_traffic(cls, road, surface, offscreen=False):
        """
            Display the road vehicles on a surface.

        :param road: the road to be displayed
        :param surface: the pygame surface
        """
        for v in road.vehicles:
            VehicleGraphics.display(v, surface, offscreen=offscreen)

class WorldSurface(pygame.Surface):
    """
        A pygame Surface implementing a local coordinate system so that we can move and zoom in the displayed area.
    """
    BLACK = (0, 0, 0)
    GREY = (100, 100, 100)
    GREEN = (50, 200, 0)
    YELLOW = (200, 200, 0)
    WHITE = (255, 255, 255)

    #Tamaño de la pantalla, queremos que se vean las 2 rotondas
    INITIAL_SCALING = 2.5

    INITIAL_CENTERING = [0.5, 0.5]
    SCALING_FACTOR = 1.3
    MOVING_FACTOR = 0.1

    def __init__(self, size, flags, surf):
        super(WorldSurface, self).__init__(size, flags, surf)
        self.origin = np.array([0, 0])
        self.scaling = self.INITIAL_SCALING
        self.centering_position = self.INITIAL_CENTERING

    def pix(self, length):
        """
            Convert a distance [m] to pixels [px].

        :param length: the input distance [m]
        :return: the corresponding size [px]
        """
        return int(length * self.scaling)

    def pos2pix(self, x, y):
        """
            Convert two world coordinates [m] into a position in the surface [px]

        :param x: x world coordinate [m]
        :param y: y world coordinate [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pix(x - self.origin[0]), self.pix(y - self.origin[1])

    def vec2pix(self, vec):
        """
             Convert a world position [m] into a position in the surface [px].
        :param vec: a world position [m]
        :return: the coordinates of the corresponding pixel [px]
        """
        return self.pos2pix(vec[0], vec[1])

    #def move_display_window_to(self, position):
    def move_display_window_to(self):
        """
            Set the origin of the displayed area to center on a given world position.
        :param position: a world position [m]
        """
        #Modifica esto para que no se mueva la pantalla!
        """ self.origin = position - np.array(
            [self.centering_position[0] * self.get_width() / self.scaling,
             self.centering_position[1] * self.get_height() / self.scaling]) """
        self.origin = [-150, -260]  #X and Y position of the window

    def handle_event(self, event):
        """
            Handle pygame events for moving and zooming in the displayed area.

        :param event: a pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_l:
                self.scaling *= 1 / self.SCALING_FACTOR
            if event.key == pygame.K_o:
                self.scaling *= self.SCALING_FACTOR
            if event.key == pygame.K_m:
                self.centering_position[0] -= self.MOVING_FACTOR
            if event.key == pygame.K_k:
                self.centering_position[0] += self.MOVING_FACTOR

class VehicleGraphics(object):
    RED = (255, 100, 100)
    GREEN = (50, 200, 0)
    BLUE = (100, 200, 255)
    YELLOW = (200, 200, 0)
    YELLOW_EGO = (245,255,0)
    BLACK = (60, 60, 60)
    PURPLE = (200, 0, 150)
    DEFAULT_COLOR = YELLOW
    EGO_COLOR = YELLOW_EGO

    @classmethod
    def display(cls, vehicle, surface, transparent=False, offscreen=False):
        """
            Display a vehicle on a pygame surface.

            The vehicle is represented as a colored rotated rectangle.

        :param vehicle: the vehicle to be drawn
        :param surface: the surface to draw the vehicle on
        :param transparent: whether the vehicle should be drawn slightly transparent
        :param offscreen: whether the rendering should be done offscreen or not
        """
        v = vehicle
        s = pygame.Surface((surface.pix(v.LENGTH), surface.pix(v.LENGTH)), pygame.SRCALPHA)  # per-pixel alpha
        rect = (0, surface.pix(v.LENGTH) / 2 - surface.pix(v.WIDTH) / 2, surface.pix(v.LENGTH), surface.pix(v.WIDTH))
        pygame.draw.rect(s, cls.get_color(v, transparent), rect, 0)
        pygame.draw.rect(s, cls.BLACK, rect, 1)
        if not offscreen:  # convert_alpha throws errors in offscreen mode TODO() Explain why
            s = pygame.Surface.convert_alpha(s)
        h = v.heading if abs(v.heading) > 2 * np.pi / 180 else 0
        sr = pygame.transform.rotate(s, -h * 180 / np.pi)
        surface.blit(sr, (surface.pos2pix(v.position[0] - v.LENGTH / 2, v.position[1] - v.LENGTH / 2)))

    @classmethod
    def display_trajectory(cls, states, surface, offscreen=False):
        """
            Display the whole trajectory of a vehicle on a pygame surface.

        :param states: the list of vehicle states within the trajectory to be displayed
        :param surface: the surface to draw the vehicle future states on
        :param offscreen: whether the rendering should be done offscreen or not
        """
        for vehicle in states:
            cls.display(vehicle, surface, transparent=True, offscreen=offscreen)

    @classmethod
    def get_color(cls, vehicle, transparent=False):
        color = cls.DEFAULT_COLOR
        if hasattr(vehicle, "color"):
            color = vehicle.color
        elif isinstance(vehicle, LinearVehicle):
            color = cls.YELLOW
        elif isinstance(vehicle, simpleVehicle):
            color = cls.BLUE
        elif isinstance(vehicle, AgentVehicle):
            color = cls.EGO_COLOR
        if transparent:
            color = (color[0], color[1], color[2], 50)
        return color

    @classmethod
    def handle_event(cls, vehicle, event):
        """
            Handle a pygame event depending on the vehicle type

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if isinstance(vehicle, AgentVehicle):
            cls.control_event(vehicle, event)
        elif isinstance(vehicle, Vehicle):
            cls.dynamics_event(vehicle, event)

    @classmethod
    def control_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to control decisions

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                vehicle.act("FASTER")
            if event.key == pygame.K_LEFT:
                vehicle.act("SLOWER")
            if event.key == pygame.K_DOWN:
                vehicle.act("LANE_RIGHT")
            if event.key == pygame.K_UP:
                vehicle.act("LANE_LEFT")

    @classmethod
    def dynamics_event(cls, vehicle, event):
        """
            Map the pygame keyboard events to dynamics actuation

        :param vehicle: the vehicle receiving the event
        :param event: the pygame event
        """
        action = vehicle.action.copy()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 45 * np.pi / 180
            if event.key == pygame.K_LEFT:
                action['steering'] = -45 * np.pi / 180
            if event.key == pygame.K_DOWN:
                action['acceleration'] = -6
            if event.key == pygame.K_UP:
                action['acceleration'] = 5
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_RIGHT:
                action['steering'] = 0
            if event.key == pygame.K_LEFT:
                action['steering'] = 0
            if event.key == pygame.K_DOWN:
                action['acceleration'] = 0
            if event.key == pygame.K_UP:
                action['acceleration'] = 0
        if action != vehicle.action:
            vehicle.act(action)
