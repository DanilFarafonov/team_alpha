import team_alpha
import pytest
from sklearn.exceptions import ConvergenceWarning
import cv2
from sklearn.utils._param_validation import InvalidParameterError
import gradio as gr

def test_generate_image():
    # test that jpeg extension image has been generated
    with pytest.warns(DeprecationWarning):
        team_alpha.picture_gen = gr.Blocks.load(name='models/prompthero/openjourney')
        assert team_alpha.generate_image('test', False)[-4:] == '.jpg'

def test_make_palette():
    # test that colors are correctly determined on test images
    assert set(team_alpha.make_palette('test_files/1.png', 1)) == {'#ffffff'}
    assert set(team_alpha.make_palette('test_files/2.png', 2)) == {'#000000', '#ffffff'}
    assert set(team_alpha.make_palette('test_files/3.png', 3)) == {'#aded02', '#02ee08', '#02f5c5'}
    assert set(team_alpha.make_palette('test_files/4.png', 4)) == {'#99d8e9', '#00a3e7', '#0c2adc', '#7091be'}
    assert set(team_alpha.make_palette('test_files/5.png', 5)) == {'#f71b0f', '#f49940', '#fef534', '#f9ca3c', '#ec5318'}
    assert set(team_alpha.make_palette('test_files/6.png', 6)) == {'#0383f5', '#f80e19', '#ed03dc', '#fa0c83', '#b80bfa', '#0a03f5'}

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

