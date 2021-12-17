import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Window 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1

import Backend 1.0


ColumnLayout {
    property var save_model: true;
    property alias pbar : pbar;

    RowLayout {
        spacing : 5

        Button {
            id : btn_reduce_model
            text : 'Reduce Model'
            font.capitalization : Font.Capitalize

            onReleased : {
                con.reduce_model();
            }
        }

        Button {
            id : btn_compute
            text : 'Compute'
            font.capitalization : Font.Capitalize

            onReleased : {
                con.compute();
            }
        }

        Button {
            id : btn_save_model
            text : 'Save Model'
            font.capitalization : Font.Capitalize

            onReleased : {
                save_file_dialog.nameFilters = ['npz (*.npz)', 'All files (*.*)'];
                save_model = true;
                save_file_dialog.open();
            }
        }

        Button {
            id : btn_save_result
            text : 'Save Result'
            font.capitalization : Font.Capitalize

            onReleased : {
                save_file_dialog.nameFilters = ['csv (*.csv)', 'All files (*.*)'];
                save_model = false;
                save_file_dialog.open();
            }
        }
    }


    Pane {
        Material.elevation : 2
        Layout.fillWidth : true
        Layout.fillHeight : true
        padding : 0

        ColumnLayout {
            anchors.fill : parent
            spacing : 0

            ProgressBar {
                id : pbar
                objectName : 'pbar'
                Layout.fillWidth : true
            }

            FigureCanvas {
                id : plot
                objectName : 'simulation_plot'
                dpi_ratio : Screen.devicePixelRatio

                Layout.fillWidth : true
                Layout.fillHeight : true
            }
        }
    }

    FileDialog {
        id : save_file_dialog

        // folder : StandardPaths.standardLocations(StandardPaths.DocumentsLocation)[0]
        fileMode : FileDialog.SaveFile

        nameFilters : ['npz (*.npz)', 'All files (*.*)']

        onAccepted : {
            if (save_model) {
                con.save_model(file);
            } else {
                con.save_results(file);
            }
        }
    }

    function update_button_hightlights(has_matrix, has_model, has_result) {
        btn_reduce_model.highlighted = has_matrix && (! has_model)
        btn_compute.highlighted = has_model
        btn_save_model.highlighted = has_model
        btn_save_result.highlighted = has_result
    }
}
