import pygame
import math
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation
from matplotlib import style
from utils import scale_image, blit_rotate_center, blit_text_center
MAX_VEL = 3
MIN_VEL = 1.5
pygame.font.init()

BLOCK = scale_image(pygame.image.load("imgs/back.png"), 3)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 0.9)
CROSS = scale_image(pygame.image.load("imgs/cross.png"), 0.01)

UPP = scale_image(pygame.image.load("imgs/fast_up.png"), 0.3)
LEFT = scale_image(pygame.image.load("imgs/left_arrow.png"), 0.3)
RIGHT = scale_image(pygame.image.load("imgs/right_arrow.png"), 0.3)
UP = scale_image(pygame.image.load("imgs/up_arrow.png"), 0.3)


TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 0.9)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = pygame.image.load("imgs/finish.png")
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (130, 250)

RED_CAR = scale_image(pygame.image.load("imgs/my_car.png"), 0.01)


WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Racing Game!")

MAIN_FONT = pygame.font.SysFont("comicsans", 44)

FPS = 70




class PlayerCar:
    def __init__(self, max_vel, rotation_vel):
        self.odom = 0
        self.img = RED_CAR
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = (170,200)
        self.acceleration = 0.1
    def draw(self, win):
        
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        self.lidar()
    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        
        self.move()
    def move_backward(self):
        self.vel = min(self.vel - self.acceleration, self.max_vel)
        self.vel = max(1,self.vel)
        self.move()
    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration/3, 0)
        self.move()

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel
        if self.angle > 360 or self.angle < -360:
            self.angle = 0
    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        
        return poi

    def move(self):
        self.vel = max(self.vel ,MIN_VEL)
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel
        self.odom += abs(vertical) + abs(horizontal)
        self.y -= vertical
        self.x -= horizontal
        


    def lidar(self):
        collision_point_x =  collision_point2_x = collision_point3_x = collision_point4_x = collision_point5_x= collision_point6_x = collision_point7_x= self.x 
        collision_point_y= collision_point2_y = collision_point3_y = collision_point4_y= collision_point5_y = collision_point6_y= collision_point7_y = self.y

        while not TRACK_BORDER_MASK.get_at((round(collision_point_x) ,round(collision_point_y))) and math.sqrt((collision_point_x-self.x)**2 + (collision_point_y-self.y)**2) <150 : 
            collision_point_x += math.cos(math.radians(self.angle+90))
            collision_point_y += math.sin(math.radians(self.angle-90))
        while not TRACK_BORDER_MASK.get_at((round(collision_point2_x) ,round(collision_point2_y))) and math.sqrt((collision_point2_x-self.x)**2 + (collision_point2_y-self.y)**2) <150: 
            collision_point2_x += math.cos(math.radians(self.angle+120))
            collision_point2_y += math.sin(math.radians(self.angle-60))
        while not TRACK_BORDER_MASK.get_at((round(collision_point3_x) ,round(collision_point3_y)))  and math.sqrt((collision_point3_x-self.x)**2 + (collision_point3_y-self.y)**2) <150: 
            collision_point3_x += math.cos(math.radians(self.angle+60))
            collision_point3_y += math.sin(math.radians(self.angle-120))
        
        while not TRACK_BORDER_MASK.get_at((round(collision_point4_x) ,round(collision_point4_y)))  and math.sqrt((collision_point4_x-self.x)**2 + (collision_point4_y-self.y)**2) <150: 
            collision_point4_x += math.cos(math.radians(self.angle+30))
            collision_point4_y += math.sin(math.radians(self.angle-150))
        while not TRACK_BORDER_MASK.get_at((round(collision_point5_x) ,round(collision_point5_y)))  and math.sqrt((collision_point5_x-self.x)**2 + (collision_point5_y-self.y)**2) <150: 
            collision_point5_x += math.cos(math.radians(self.angle+150))
            collision_point5_y += math.sin(math.radians(self.angle-30))
        while not TRACK_BORDER_MASK.get_at((round(collision_point6_x) ,round(collision_point6_y)))  and math.sqrt((collision_point6_x-self.x)**2 + (collision_point6_y-self.y)**2) <150: 
            collision_point6_x += math.cos(math.radians(self.angle+180))
            collision_point6_y += math.sin(math.radians(self.angle))
        while not TRACK_BORDER_MASK.get_at((round(collision_point7_x) ,round(collision_point7_y)))  and math.sqrt((collision_point7_x-self.x)**2 + (collision_point7_y-self.y)**2) <150: 
            collision_point7_x += math.cos(math.radians(self.angle))
            collision_point7_y += math.sin(math.radians(self.angle-180))
        
        
        WIN.blit(CROSS, (collision_point_x, collision_point_y))
        WIN.blit(CROSS, (collision_point2_x, collision_point2_y))
        WIN.blit(CROSS, (collision_point3_x, collision_point3_y))
        WIN.blit(CROSS, (collision_point4_x, collision_point4_y))
        WIN.blit(CROSS, (collision_point5_x, collision_point5_y))
        WIN.blit(CROSS, (collision_point6_x, collision_point6_y))
        WIN.blit(CROSS, (collision_point7_x, collision_point7_y))
        data_1 = math.sqrt((collision_point_x-self.x)**2 + (collision_point_y-self.y)**2)
        data_2 = math.sqrt((collision_point2_x-self.x)**2 + (collision_point2_y-self.y)**2)
        data_3 = math.sqrt((collision_point3_x-self.x)**2 +(collision_point3_y-self.y)**2)
        data_4 = math.sqrt((collision_point4_x-self.x)**2 + (collision_point4_y-self.y)**2)
        data_5 = math.sqrt((collision_point5_x-self.x)**2 + (collision_point5_y-self.y)**2)
        data_6 = math.sqrt((collision_point6_x-self.x)**2 + (collision_point6_y-self.y)**2)
        data_7 = math.sqrt((collision_point7_x-self.x)**2 + (collision_point7_y-self.y)**2)
        
        return(self.sigmoid2([data_1,data_2,data_3,data_4,data_5,data_6,data_7]))


    def sigmoid2(self,data):
        rt = []
        for i in range(len(data)):
            d = 1/(1+math.exp((-data[i]/15)+5))
            rt.append(d)
        return rt
    
        
    def get_data(self,player_car):
       
        images = [ (TRACK, (0, 0)),
                (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
        

        self.check = False
        run = True

        data_x = np.array(player_car.lidar())
        data_y = np.array(move_player(player_car))

        while run:
            
            data_x = np.vstack((data_x,player_car.lidar()))
            data_y = np.vstack((data_y,move_player(player_car)))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                    

            if player_car.collide(TRACK_BORDER_MASK) != None:

                return data_x, data_y
                
            draw(WIN, images, player_car)
            pygame.display.update()
            
def test_ai(weights_hidden2_output,weights_hidden1_hidden2,weights_input_hidden1,player_car):
    
    run = True
    images = [ (TRACK, (0, 0)),
                (FINISH, FINISH_POSITION), (TRACK_BORDER, (0, 0))]
    while run:
        test_hidden1_input = np.dot(player_car.lidar(), weights_input_hidden1)
        test_hidden1_output = (test_hidden1_input)
        test_hidden2_input = np.dot(test_hidden1_output, weights_hidden1_hidden2)
        test_hidden2_output = sigmoid(test_hidden2_input)
        test_output_input = np.dot(test_hidden2_output, weights_hidden2_output)
        test_output_output = sigmoid(test_output_input)
        d1,d2,d3 = test_output_output
        print(d1,d2,d3)
        if round(d1) :
            print("left")
            player_car.rotate(left=True)
            
        if round(d2):
            print("right")
            player_car.rotate(right=True)
            
        if round(d3):
            print("fowrward")
            
            player_car.move_forward()
        else:
            player_car.reduce_speed()
        draw(WIN, images, player_car)
        pygame.display.update()
        if player_car.collide(TRACK_BORDER_MASK) != None:
                
            run = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
    save = input("save the model ? yes/no")
    if save == 'yes':
        save_to_file(weights_hidden2_output,weights_hidden1_hidden2,weights_input_hidden1)
    

                

def save_to_file(weights_hidden2_output,weights_hidden1_hidden2,weights_input_hidden1): 
    np.savez('./Saved_model.npz', weights_hidden2_output=weights_hidden2_output,weights_hidden1_hidden2=weights_hidden1_hidden2,weights_input_hidden1=weights_input_hidden1 )
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)
    
    
def train_data(x,y,player_car):
    input_layer_size = 7
    threshold = 0.1
    hidden_layer1_size = 10
    hidden_layer2_size = 10
    output_layer_size = 3
    data_x = np.array(x)
    data_y = np.array(y)
    data_to_shuffle = np.concatenate((data_x,data_y), axis=1)
    np.random.shuffle(data_to_shuffle)
    
    print(data_to_shuffle.shape)
    data_x = data_to_shuffle[:,:7]
    data_y = data_to_shuffle[:,7:]
    weights_input_hidden1 = 2 * np.random.random((input_layer_size, hidden_layer1_size)) - 1
    weights_hidden1_hidden2 = 2 * np.random.random((hidden_layer1_size, hidden_layer2_size)) - 1
    weights_hidden2_output = 2 * np.random.random((hidden_layer2_size, output_layer_size)) - 1


    learning_rate = 0.00001
    epochs = 8000
    eror_graph = []
    start_time = time.time()
    for epoch in range(epochs):
        for test_data,val_data in zip(data_x,data_y):
            
            input_layer =  test_data.reshape(1,7)
        
            hidden_layer1_input = np.dot(input_layer, weights_input_hidden1)
            
            hidden_layer1_output = (hidden_layer1_input)
            hidden2_layer_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
            hidden_layer2_output = sigmoid(hidden2_layer_input)
            output_layer_input = np.dot(hidden_layer2_output, weights_hidden2_output)
            output_layer_output = sigmoid(output_layer_input)

            
            error = val_data.reshape(-1, output_layer_size) - output_layer_output

            

            d_output = error * sigmoid_derivative(output_layer_output)
            error_hidden2 = d_output.dot(weights_hidden2_output.T)
            d_hidden2 = error_hidden2 * sigmoid_derivative(hidden_layer2_output)
            error_hidden1 = d_hidden2.dot(weights_hidden1_hidden2.T)
            d_hidden1 = error_hidden1 * sigmoid_derivative(hidden_layer1_output)

            weights_hidden2_output += hidden_layer2_output.T.dot(d_output) * learning_rate
            weights_hidden1_hidden2 += hidden_layer1_output.T.dot(d_hidden2) * learning_rate
            weights_input_hidden1 += input_layer.T.dot(d_hidden1) * learning_rate
            if time.time() - start_time > 10:

                print("f----",hidden_layer2_output, "d----",hidden2_layer_input )
                print("Training ...  %",epoch/epochs*100,"   Error:",error)
                start_time = time.time()

        eror_graph.append(np.sum(error))
    plt.plot(eror_graph,label='error')

    plt.xlabel('epoch')
    plt.ylabel('error')
    plt.legend()  
    plt.title('graph ')
    plt.show()
    test_ai(weights_hidden2_output,weights_hidden1_hidden2,weights_input_hidden1,player_car)




def draw(win, images, player_car):
    for img, pos in images:
        win.blit(img, pos)
    player_car.draw(win)


def move_player(player_car):
    keys = pygame.key.get_pressed()
    moved = False
    d1 = d2 = d3 = 0
    if keys[pygame.K_a]:
        player_car.rotate(left=True)
        d1 = 1
        d2=0
    if keys[pygame.K_d]:
        player_car.rotate(right=True)
        d2 = 1
        d1 = 0

    if keys[pygame.K_w]:
        
        player_car.move_forward()
        d3 = 1
    else:
        player_car.reduce_speed()
        d3 = 0
    return d1,d2,d3


def train_and_test(play):
    
    genoms =1
    data_x , data_y  = play.get_data(play)
    for genom in range(genoms) :
        play = PlayerCar(MAX_VEL,4)
        data  = play.get_data(play)
        if len(data[0]) > 100:
            

            data_x = np.concatenate((data_x,data[0]),axis=0)
            data_y = np.concatenate((data_y,data[1]),axis=0)
            
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            
    play1 = PlayerCar(MAX_VEL,5)
    pygame.display.update()
    print("Training ...")
    train_data(data_x,data_y,play1)
if __name__ == '__main__':
    play = PlayerCar(MAX_VEL,4)
    loaded_data = np.load('./Saved_model.npz')


    weights_hidden2_output = loaded_data['weights_hidden2_output']
    weights_hidden1_hidden2 = loaded_data['weights_hidden1_hidden2']
    weights_input_hidden1 = loaded_data['weights_input_hidden1']
    train_and_test(play)
    #test_ai(weights_hidden2_output, weights_hidden1_hidden2,weights_input_hidden1,play)
    

