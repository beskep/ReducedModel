import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15
import QtQuick.Layouts 1.15


Item {
    id : root

    GridLayout {
        anchors.fill : parent
        flow : GridLayout.TopToBottom
        columnSpacing : 5
        rowSpacing : 5

        CustomBox {
            title : 'Computation'

            ColumnLayout {
                anchors.fill : parent

                OptionItem {
                    label.text : 'Reduced Model Order'
                    value : '10'
                    option_id : 'order'
                    validator : IntValidator {}
                }
                OptionItem {
                    label.text : 'Time Steps'
                    value : '100'
                    option_id : 'time steps'
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
            }
        }

        CustomBox {
            title : 'Internal condition'

            ColumnLayout {
                anchors.fill : parent

                OptionItem {
                    label.text : 'Fluid temperature'
                    value : '20.0'
                    unit.text : '℃'
                    option_id : 'internal fluid temperature'
                }
                OptionItem {
                    label.text : 'Max temperature'
                    value : '25.0'
                    unit.text : '℃'
                    option_id : 'internal max temperature'
                }
                OptionItem {
                    label.text : 'Min temperature'
                    value : '15.0'
                    unit.text : '℃'
                    option_id : 'internal min temperature'
                }
            }
        }

        CustomBox {
            title : 'External condition'

            ColumnLayout {
                anchors.fill : parent

                OptionItem {
                    label.text : 'Fluid temperature'
                    value : '10.0'
                    unit.text : '℃'
                    option_id : 'external fluid temperature'
                }
                OptionItem {
                    label.text : 'Max temperature'
                    value : '15.0'
                    unit.text : '℃'
                    option_id : 'external max temperature'
                }
                OptionItem {
                    label.text : 'Min temperature'
                    value : '5.0'
                    unit.text : '℃'
                    option_id : 'external min temperature'
                }
            }
        }
    }
}
