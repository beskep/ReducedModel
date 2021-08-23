import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15

Pane {
    height : 32
    horizontalPadding : 20
    verticalPadding : 0

    property alias status_text : status_text
    property var icon_blank: '../../resource/check_box_outline_blank_black_24dp.svg'
    property var icon_check: '../../resource/check_box_black_24dp.svg'

    background : Rectangle {
        color : '#E0E0E0'
    }

    RowLayout {
        anchors.fill : parent

        RowLayout {
            Label {
                text : 'Matrix'
                font.pointSize : 12
                Layout.alignment : Qt.AlignLeft | Qt.AlignBottom
            }

            Image {
                id : matrix_icon
                source : icon_blank
                Layout.preferredHeight : 22
                Layout.preferredWidth : 22
                Layout.alignment : Qt.AlignLeft | Qt.AlignBottom
            }
        }

        ToolSeparator {
            Layout.fillHeight : true
            Layout.alignment : Qt.AlignLeft
        }

        RowLayout {
            Label {
                text : 'Model'
                font.pointSize : 12
                Layout.alignment : Qt.AlignLeft | Qt.AlignBottom
            }

            Image {
                id : model_icon
                source : icon_blank
                Layout.preferredHeight : 22
                Layout.preferredWidth : 22
                Layout.alignment : Qt.AlignLeft | Qt.AlignBottom
            }
        }

        ToolSeparator {
            Layout.fillHeight : true
        }

        Label {
            id : status_text

            Layout.alignment : Qt.AlignVCenter
            font.pointSize : 12
            color : "#212121"

            text : '리덕션 수치모델 기반 창호시공부위 하자 평가 프로그램'
        }

        Label {
            Layout.fillWidth : true
        }
    }

    function update_model_status_icons(has_matrix, has_model) {
        if (has_matrix) {
            matrix_icon.source = icon_check
        } else {
            matrix_icon.source = icon_blank
        }

        if (has_model) {
            model_icon.source = icon_check
        } else {
            model_icon.source = icon_blank
        }
    }
}
