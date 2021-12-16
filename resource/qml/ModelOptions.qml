import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1

import 'custom'


Item {
    id : root

    ColumnLayout {
        anchors.fill : parent

        OptionItem {
            label.text : 'Reduced Model Order'
            value : '10'
            option_id : 'order'
            validator : IntValidator {}
        }
        OptionItem {
            label.text : 'Δt'
            value : '3600.0'
            unit.text : 'sec'
            option_id : 'deltat'
        }
        OptionItem {
            label.text : 'Initial temperature'
            value : '0.0'
            unit.text : '℃'
            option_id : 'initial temperature'
        }
        // TODO reference models 선택했을 때 air temperature 비활성화
        OptionItem {
            label.text : 'Internal air temperature'
            value : '0.0'
            unit.text : '℃'
            option_id : 'internal air temperature'
        }
        OptionItem {
            label.text : 'External air temperature'
            value : '0.0'
            unit.text : '℃'
            option_id : 'external air temperature'
        }

        RowLayout {
            id : log
            TextField {
                id : log_path
                Layout.fillWidth : true
                placeholderText : 'Temperature Log Path'

                onEditingFinished : {
                    con.set_option('temperature log path|' + text);
                }
            }
            Button {
                highlighted : true
                font.capitalization : Font.Capitalize
                text : 'Load'
                onReleased : file_dialog.open()
            }
        }
    }

    FileDialog {
        id : file_dialog
        nameFilters : ['csv (*.csv)', 'All files (*)']
        fileMode : FileDialog.OpenFile

        onAccepted : {
            log_path.text = file.toString().replace('file:///', '');
            log_path.editingFinished();
        }
    }
}
