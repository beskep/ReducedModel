from multiprocessing import freeze_support
import os
import sys

from rm.interface.init import init_project

init_project()

# pylint: disable=wrong-import-position, no-name-in-module
from loguru import logger
from matplotlib_backend_qtquick.backend_qtquickagg import FigureCanvas
from matplotlib_backend_qtquick.qt_compat import QtCore
from matplotlib_backend_qtquick.qt_compat import QtGui
from matplotlib_backend_qtquick.qt_compat import QtQml
from matplotlib_backend_qtquick.qt_compat import QtQuick

from rm import utils
from rm.interface.controller import Controller
from rm.interface.plot_controller import OptimizationPlotController
from rm.interface.plot_controller import SimulationPlotController


def main(log_level=None):
  freeze_support()

  utils.set_logger(log_level)

  cfg_path = utils.DIR.RESOURCE.joinpath('qtquickcontrols2.conf')
  qml_path = utils.DIR.RESOURCE.joinpath('qml/main.qml')
  for p in [cfg_path, qml_path]:
    p.stat()

  # graphic setting
  QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
  os.environ['QT_QUICK_CONTROLS_CONF'] = cfg_path.as_posix()

  # app
  app = QtGui.QGuiApplication(sys.argv)
  app.setOrganizationName('Inha University & Mirae Environment Plan')
  app.setOrganizationDomain('-')

  engine = QtQml.QQmlApplicationEngine()
  context = engine.rootContext()

  # set controller
  controller = Controller()
  spc = SimulationPlotController()
  opc = OptimizationPlotController()
  context.setContextProperty('con', controller)
  # context.setContextProperty('plot_con', spc)

  # register figure canvas
  QtQml.qmlRegisterType(FigureCanvas, 'Backend', 1, 0, 'FigureCanvas')

  # load qml
  engine.load(qml_path.as_posix())

  root_objects = engine.rootObjects()
  if not root_objects:
    logger.error('Failed to load QML')
    sys.exit()

  # set controllers
  win: QtGui.QWindow = root_objects[0]

  scanvas = win.findChild(QtCore.QObject, 'simulation_plot')
  spc.update_with_canvas(scanvas)
  spc.set_app(app)

  ocanvas = win.findChild(QtCore.QObject, 'optimization_plot')
  opc.update_with_canvas(ocanvas)
  opc.set_app(app)

  controller.set_window(win)
  controller.set_plot_controller(spc, opc)

  # run
  sys.exit(app.exec_())


if __name__ == '__main__':
  main()
