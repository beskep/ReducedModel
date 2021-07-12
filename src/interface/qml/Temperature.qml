import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Window 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1

import Backend 1.0


Item {
    id : root

    FileDialog {
        id : file_dialog

        property var location: 'init'

        nameFilters : ['All files (*.*)']
        folder : StandardPaths.standardLocations(StandardPaths.DocumentsLocation)[0]

        // for Qt.labs.platform
        fileMode : FileDialog.OpenFile

        onAccepted : {
            var file_clean = file.toString().replace('file:///', '');
            con.read_temperature(location + '|' + file_clean);

            if (location == 'interior') {
                interior_path.text = file_clean
            } else if (location == 'exterior') {
                exterior_path.text = file_clean
            } else {
                con.log('ERROR|FileDialog location error')
            }
        }
    }

    RowLayout {
        anchors.fill : parent

        CustomBox {
            title : 'Thermal Image'
            Layout.preferredWidth : 250
            Layout.fillHeight : true

            // TODO target nodes 개수에 따라 동적으로 설정
            ColumnLayout {
                width : parent.width

                IR {
                    title : 'Temperature on mode 1'
                }
                IR {
                    title : 'Temperature on mode 2'
                }
                IR {
                    title : 'Temperature on mode 3'
                }
            }
        }

        CustomBox {
            title : 'Logging Files'
            Layout.fillWidth : true
            Layout.fillHeight : true

            ColumnLayout {
                width : parent.width
                Layout.fillHeight : true

                RowLayout {
                    Layout.fillWidth : true

                    TextField {
                        text : 'Interior'
                        readOnly : true
                        // TODO width 설정
                    }
                    TextField {
                        id : interior_path
                        objectName : 'interior_path'

                        Layout.fillWidth : true
                    }
                    Button {
                        text : 'Select'

                        onReleased : {
                            file_dialog.location = 'interior';
                            file_dialog.open();
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth : true

                    TextField {
                        text : 'Exterior'
                    }
                    TextField {
                        id : exterior_path
                        objectName : 'exterior_path'

                        Layout.fillWidth : true
                    }
                    Button {
                        text : 'Select'

                        onReleased : {
                            file_dialog.location = 'exterior';
                            file_dialog.open();
                        }
                    }
                }

                // TODO logged temperature 그래프?
                // Pane {
                //     Material.elevation : 1
                //     Layout.fillWidth : true
                //     Layout.fillHeight : true
                //     Layout.preferredHeight : 400

                //     FigureCanvas {
                //         id : plot
                //         objectName : 'plot'
                //         dpi_ratio : Screen.devicePixelRatio

                //         anchors.fill : parent
                //     }
                // }
            }
        }
    }
}
