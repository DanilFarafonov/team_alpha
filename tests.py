import team_alpha
import pytest
from sklearn.exceptions import ConvergenceWarning
import cv2
from sklearn.utils._param_validation import InvalidParameterError
import gradio as gr

def test_generate_image():
    # test that jpeg extension image has been generated
    team_alpha.picture_gen = gr.Blocks.load(name='models/prompthero/openjourney')
    assert team_alpha.generate_image('test', False)[-4:] == '.jpg'

def test_make_palette():
    # test that colors are correctly determined on test images
    assert set(team_alpha.make_palette('test_files/1.png', 1)) == {'#ffffff'}
    assert set(team_alpha.make_palette('test_files/2.png', 2)) == {'#000000', '#ffffff'}
    assert set(team_alpha.make_palette('test_files/3.png', 3)) == {'#6cb368', '#47fa38', '#52faa6'}
    assert set(team_alpha.make_palette('test_files/4.png', 4)) == {'#3f48cc', '#00a2e8', '#7092be', '#99d9ea'}
    assert set(team_alpha.make_palette('test_files/5.png', 5)) == {'#ed1b24', '#f03f2b', '#f8b823', '#f47d26', '#fff200'}
    assert set(team_alpha.make_palette('test_files/6.png', 6)) == {'#ed1c24', '#1d2ded', '#ed1db9', '#761feb', '#c41ded', '#1d85ed'}

    # when request more colors than are in the image
    with pytest.warns(ConvergenceWarning):
        team_alpha.make_palette('test_files/1.png', 2)

    # when instead of an image we transfer a file of another extension or nothing
    with pytest.raises(cv2.error):
        team_alpha.make_palette('', 1)
        team_alpha.make_palette('test_files/text.txt', 1)

    # when setting an unplanned palette size
    with pytest.raises(InvalidParameterError):
        team_alpha.make_palette('test_files/1.png', 0)
        team_alpha.make_palette('test_files/1.png', 2.5)
        team_alpha.make_palette('test_files/1.png', -3)
        team_alpha.make_palette('test_files/1.png', '2')

def test_color_table():
    colors = ['#000000', '#ffffff']
    rows = team_alpha.color_table(colors).split('\n')

    # test that the output palette displays the color codes passed to the function
    # the output number of color codes corresponds to the number of colors passed to the function
    counter = 0
    for color in colors:
        if color in rows[0]:
            counter += 1
    assert counter == len(colors)

    # test that the output palette displays the colors passed to the function
    # test that the displayed number of colors corresponds to the number of colors sent to the function
    counter = 0
    for color in colors:
        if color in rows[2]:
            counter += 1
    assert counter == len(colors)

def test_generate_color_table():
    # if there are no input parameters, then the predefined parameters are used
    team_alpha.generate_color_table()

