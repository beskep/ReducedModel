import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15
import Qt.labs.platform 1.1

import 'custom'


Item {
    id : root

    property var at_enabled: false;
    property var order_enabled: false;

    ColumnLayout {
        anchors.fill : parent

        OptionItem {
            id : _at_int
            label.text : 'Internal Air Temperature'
            value : '0.0'
            unit.text : '℃'
            option_id : 'internal air temperature'
            tooltip : '실내 air temperature (행렬 파일 입력 시 필요)'
            enabled : at_enabled
        }
        OptionItem {
            id : _at_ext
            label.text : 'External Air Temperature'
            value : '0.0'
            unit.text : '℃'
            option_id : 'external air temperature'
            tooltip : '실외 air temperature (행렬 파일 입력 시 필요)'
            enabled : at_enabled
        }

        OptionItem {
            label.text : 'Reduced Model Order'
            value : '10'
            option_id : 'order'
            tooltip : '모델 리덕션 목표 차수'
            validator : IntValidator {}
            enabled : order_enabled
        }

        Text {}

        OptionItem {
            label.text : 'Δt'
            value : '3600.0'
            unit.text : 'sec'
            option_id : 'deltat'
            tooltip : '시뮬레이션 시간 간격'
        }
        OptionItem {
            label.text : 'Initial Temperature'
            value : '0.0'
            unit.text : '℃'
            option_id : 'initial temperature'
            tooltip : '초기 온도'
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

                ToolTip.visible : hovered
                ToolTip.text : '실내외 온도 실측 결과 불러오기'

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
