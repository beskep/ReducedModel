import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Dialogs 1.3
import QtQuick.Layouts 1.15


Item {
    property alias index : _index.text
    property alias day : _day
    property alias time : _time
    property alias point : _point
    property alias temperature : _temperature

    Layout.preferredHeight : 45
    Layout.fillHeight : true

    function temperature_measurement() {
        con.temperature_measurement( //
            _index.text, _day.text, _time.text, _point.text, _temperature.text)
    }

    Component.onCompleted : temperature_measurement()

    RowLayout {
        anchors.fill : parent
        spacing : 5

        Label {
            id : _index
            text : '00.'
        }

        RowLayout {

            TextField {
                id : _day

                implicitWidth : 30
                selectByMouse : true
                horizontalAlignment : TextField.AlignHCenter
                color : enabled ? '#000000' : '#00000000'

                validator : IntValidator {}

                text : '0'
                ToolTip.visible : hovered
                ToolTip.text : '시뮬레이션 시작 시점 (1 day 00:00:00) 기준 측정 시간'

                onEditingFinished : temperature_measurement()
            }

            Label {
                text : 'day'
                color : _day.enabled ? '#000000' : '#00000000'
            }

            TextField {
                id : _time

                implicitWidth : 80
                selectByMouse : true
                horizontalAlignment : TextField.AlignHCenter

                inputMask : '99:99:99'
                inputMethodHints : Qt.ImhDigitsOnly
                validator : RegExpValidator {
                    regExp : /^([0-1]?[0-9]|2[0-3]):([0-5][0-9]):[0-5][0-9]$ /
                }
                ToolTip.visible : hovered
                ToolTip.text : '시뮬레이션 시작 시점 (1 day 00:00:00) 기준 측정 시간'

                enabled : _day.enabled
                color : enabled ? '#000000' : '#00000000'

                text : '00:00:00'

                onTextChanged : temperature_measurement()
            }
        }

        TextField {
            id : _point
            implicitWidth : 70
            readOnly : true
            text : ''
        }

        RowLayout {
            TextField {
                id : _temperature

                implicitWidth : 50
                selectByMouse : true
                horizontalAlignment : TextField.AlignRight

                validator : DoubleValidator {}

                text : '0.0'

                onEditingFinished : temperature_measurement()
            }
            Label {
                text : '℃'
            }
        }
    }
}
