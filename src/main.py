import os
import sys
from multiprocessing import freeze_support

import utils

from loguru import logger
from PyQt5 import QtCore, QtGui, QtQml

from interface import Controller, PlotController
from matplotlib_backend_qtquick import FigureCanvasQtQuickAgg


def main():
  freeze_support()

  utils.set_logger()

  cfg_path = utils.SRC_DIR.joinpath('interface/qtquickcontrols2.conf')
  qml_path = utils.SRC_DIR.joinpath('interface/qml/main.qml')
  font_paths = [
      utils.SRC_DIR.joinpath('resource/Spoqa Han Sans Neo {}.otf'.format(x))
      for x in ['Bold', 'Light', 'Regular']
  ]

  for p in [cfg_path, qml_path] + font_paths:
    if not p.exists():
      raise FileNotFoundError(p)

  # graphic setting
  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
  os.environ['QT_QUICK_CONTROLS_CONF'] = cfg_path.as_posix()

  # for p in font_paths:
  #   QtGui.QFontDatabase.addApplicationFont(
  #       ':/' + p.relative_to(utils.ROOT_DIR).as_posix())

  # app
  app = QtGui.QGuiApplication(sys.argv)
  app.setOrganizationName('Inha University & Mirae Environment Plan')
  app.setOrganizationDomain('-')

  engine = QtQml.QQmlApplicationEngine()
  context = engine.rootContext()

  # set controller
  controller = Controller()
  plot_controller = PlotController()
  context.setContextProperty('con', controller)
  context.setContextProperty('plot_con', plot_controller)

  # register figure canvas
  QtQml.qmlRegisterType(FigureCanvasQtQuickAgg, 'Backend', 1, 0, 'FigureCanvas')

  # load qml
  engine.load(qml_path.as_posix())

  root_objects = engine.rootObjects()
  if not root_objects:
    logger.error('Failed to load QML')
    sys.exit()

  # set plot controller
  win: QtGui.QWindow = root_objects[0]
  canvas = win.findChild(QtCore.QObject, 'plot')
  plot_controller.update_with_canvas(canvas)
  plot_controller.set_app(app)

  controller.set_window(win)
  controller.set_plot_controller(plot_controller)

  # run
  sys.exit(app.exec_())


if __name__ == '__main__':
  main()
